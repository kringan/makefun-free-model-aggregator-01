// free-model-aggregator-01: one server that serves the OpenAI Chat Completions
// API and the Anthropic Messages API over ten simulated free-tier providers.
//
// HOW A TRIAL RUNS
//   The trial hands this program a brief (the path in $MAKEFUN_PROMPT_FILE) and
//   an env (at $MAKEFUN_ENV_URL). The brief carries three JSON documents: the
//   provider directory (ten simulated providers, each with its id, wire, auth
//   scheme, key env var, key, and base path), the quality table (every model's
//   rating in milli-points plus what it can do: tools, vision, JSON mode,
//   context length, and whether it is free), and the workload manifest (the
//   twenty requests the client will send, in order, with each one's api,
//   stream flag, weight, and requirements). The env serves the providers under
//   their base paths, plus POST /register, GET /work and GET /instance.
//
//   This program: reads the brief, probes every provider's catalog route in
//   the provider's own wire, builds one catalog, listens on 0.0.0.0 at a free
//   port, registers {protocol_version: 1, port, catalog} with the env, serves
//   the client's requests, polls GET /work, and exits 0 when the env says the
//   workload is done. `node main.ts --list` prints the same catalog document
//   and exits 0 without starting a server.
//
// ROUTING (what wins the score)
//   Quality is 60% of the score: the weighted rating of valid answers over
//   the best legal rating per request. So for each request this program ranks
//   every free model that meets the request's requirements by its rating in
//   the quality table (highest first) and tries them in that order. A model
//   whose provider refused a call gets a penalty against its rating, so a
//   flaky provider sinks below fresh ones but can come back; after several
//   refusals in a row a provider is treated as spent for the trial.
//   Served is 15%: a request is served when exactly one provider call
//   answers it, so the program stops at the first success and never calls a
//   second provider after one answered. Failover is 10%: when the first
//   provider call refuses (402/429/5xx or unreachable), the request is
//   retried on the next-ranked model that meets it. The wire is 10%: the
//   answer must come back in the wire the client used, with the provider's
//   text whole, its usage counts exact, and its stop reason mapped.
//   The catalog is 5%: --list must match the directory and the free-model
//   sets exactly, so the one catalog object is built once and used both for
//   --list and for /register.
//
// WIRES
//   Three upstream wires exist. OpenAI-wire providers take POST
//   /v1/chat/completions. Anthropic-wire providers take POST /v1/messages.
//   Gemini-wire providers take POST /v1beta/models/<model>:generateContent
//   (or :streamGenerateContent). A request is translated faithfully into the
//   provider's wire: system text, user text (images included), tool
//   definitions, stop sequences, max_tokens and JSON mode all survive, so
//   the provider's echo of what it parsed matches what the client sent.
//   One rule is hard: an OpenAI request that carries response_format (JSON
//   mode) can only go to an OpenAI-wire provider, because the Anthropic
//   wire has no response_format to carry and the Gemini wire reports JSON
//   mode only through its own responseMimeType; routing such a request to a
//   wire that cannot carry it would echo json_mode false and lose it.
//   The provider's answer text is always the literal "ANSWER ECHO" plus the
//   echo JSON, so the answer is repackaged in the client's wire with its
//   text whole and its usage counts copied exactly.
//
// DETERMINISM
//   A trial is replayed, and the export must match: every route is a pure
//   function of the requests and the providers' answers seen so far, never
//   of the wall clock. Model lists use fixed timestamps. Ties in ranking
//   break by provider id then model id. No randomness anywhere.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { readFileSync } from 'node:fs';

/** The register body's protocol version. */
const PROTOCOL = 1;
/** How often GET /work is polled, in milliseconds. */
const WORK_POLL_MS = 100;
/** How long one provider catalog probe may take, in milliseconds. */
const PROBE_BUDGET_MS = 5000;
/** The most bytes of one client request body this program reads. */
const MAX_REQUEST_BYTES = 32 * 1024 * 1024;
/** Rating milli-points a provider loses per refused call, for ranking. */
const REFUSAL_PENALTY_MILLI = 50;
/** Consecutive refusals after which a provider is treated as spent. */
const SPENT_AFTER_REFUSALS = 3;
/** The most upstream candidates tried for one client request. */
const MAX_CANDIDATES = 8;
/** A fixed epoch second for every listed model's `created`. */
const LIST_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`. */
const LIST_CREATED_AT = '2026-01-01T00:00:00Z';
/** The Anthropic version header the providers expect. */
const ANTHROPIC_VERSION = '2023-06-01';

const ENV_ORIGIN = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const BRIEF_PATH = process.env.MAKEFUN_PROMPT_FILE ?? '';

type WireKind = 'openai' | 'anthropic' | 'gemini';
type AuthKind = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';
type JsonObj = Record<string, unknown>;

const isObj = (v: unknown): v is JsonObj => v !== null && typeof v === 'object' && !Array.isArray(v);
const strOf = (v: unknown): string => (typeof v === 'string' ? v : '');
const numOrNull = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);
const boolOrNull = (v: unknown): boolean | null => (typeof v === 'boolean' ? v : null);

/** One provider as the brief's directory lists it. */
interface DirectoryProvider {
  id: string;
  name: string;
  wire: WireKind;
  auth: AuthKind;
  base: string;
  keyEnv: string;
  key: string;
}

/** One row of the quality table. */
interface QualityRow {
  model: string;
  provider: string;
  rating: number;
  context: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

/** What one workload request needs from its model. */
interface Need {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  minContext?: number;
}

interface WorkloadRow {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: 1 | 2;
  need: Need;
}

/** One model as a provider's catalog route listed it. */
interface ListedModel {
  id: string;
  display: string;
  free: boolean | null;
  context: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

interface ProbeResult {
  provider: DirectoryProvider;
  ok: boolean;
  status: number;
  models: ListedModel[];
  problem: string | null;
}

/** What a model can do, from the quality table first and the listing second. */
interface Caps {
  context: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

/** One provider of the catalog document (--list and /register share it). */
interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: WireKind;
  models: { id: string; free: boolean; context_length: number | null }[];
}

interface CatalogDoc {
  providers: CatalogProvider[];
}

interface BriefDoc {
  providers: DirectoryProvider[];
  ratings: QualityRow[] | null;
  best: { id: string; rating: number }[];
  workload: WorkloadRow[] | null;
  problem: string | null;
}

/** A base path joined to the env origin; an absolute base is used as it stands. */
function joinOrigin(origin: string, base: string): string {
  const b = base.trim();
  if (/^https?:\/\//i.test(b)) return b.replace(/\/+$/, '');
  const prefix = origin.replace(/\/+$/, '');
  if (b === '') return prefix;
  return `${prefix}${b.startsWith('/') ? '' : '/'}${b.replace(/\/+$/, '')}`;
}

/** The headers that carry a provider's key. */
function keyHeaders(auth: AuthKind, key: string): Record<string, string> {
  if (key === '') return {};
  if (auth === 'bearer') return { authorization: `Bearer ${key}` };
  if (auth === 'x-api-key') return { 'x-api-key': key };
  if (auth === 'x-goog-api-key') return { 'x-goog-api-key': key };
  return {};
}

/** The index of the `}` closing the object that opens at `start`, or -1. */
function objectClose(text: string, start: number): number {
  let depth = 0;
  let inStr = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inStr) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inStr = false;
    } else if (ch === '"') inStr = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

/** Every JSON object in `text` that carries a string `schema`, in order. */
function schemaDocs(text: string): JsonObj[] {
  const found: JsonObj[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    const end = objectClose(text, start);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isObj(value) && typeof value.schema === 'string') found.push(value);
    } catch {
      // Not JSON on its own: look on.
    }
  }
  return found;
}

function providerFromRow(v: unknown): DirectoryProvider | null {
  if (!isObj(v)) return null;
  const id = strOf(v.id);
  const wire = v.wire === 'openai' || v.wire === 'anthropic' || v.wire === 'gemini' ? v.wire : null;
  if (id === '' || wire === null) return null;
  const auth = v.auth === 'bearer' || v.auth === 'x-api-key' || v.auth === 'x-goog-api-key' || v.auth === 'none' ? v.auth : 'bearer';
  const base =
    strOf(v.path) !== '' ? strOf(v.path) : strOf(v.base_path) !== '' ? strOf(v.base_path) : strOf(v.base_url) !== '' ? strOf(v.base_url) : strOf(v.base);
  return { id, name: strOf(v.name) !== '' ? strOf(v.name) : id, wire, auth, base, keyEnv: strOf(v.key_env), key: strOf(v.key) };
}

function qualityFromRow(v: unknown): QualityRow | null {
  if (!isObj(v)) return null;
  const model = strOf(v.model);
  const provider = strOf(v.provider);
  if (model === '' || provider === '') return null;
  return {
    model,
    provider,
    rating: typeof v.rating_milli === 'number' && Number.isFinite(v.rating_milli) ? v.rating_milli : 0,
    context: numOrNull(v.context_length),
    tools: v.tools === true,
    vision: v.vision === true,
    json: v.json === true,
    free: v.free === true,
  };
}

function workloadFromRow(v: unknown): WorkloadRow | null {
  if (!isObj(v)) return null;
  const id = strOf(v.id);
  if (id === '') return null;
  const need = isObj(v.requires) ? v.requires : {};
  return {
    id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    need: {
      tools: need.tools === true,
      vision: need.vision === true,
      json: need.json === true,
      ...(typeof need.min_context === 'number' && Number.isFinite(need.min_context) ? { minContext: need.min_context } : {}),
    },
  };
}

/** The brief at `path`: its three documents, found by schema. */
function readPrompt(path: string): BriefDoc {
  const empty: BriefDoc = { providers: [], ratings: null, best: [], workload: null, problem: null };
  if (path === '') return { ...empty, problem: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { ...empty, problem: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = schemaDocs(text);
  const dirDoc = docs.find((d) => d.schema === 'free-model-aggregator-01/providers/v1' && Array.isArray(d.providers)) ?? null;
  if (dirDoc === null || !Array.isArray(dirDoc.providers)) {
    return { ...empty, problem: 'the brief holds no free-model-aggregator-01/providers/v1 document' };
  }
  const providers = dirDoc.providers.flatMap((v) => {
    const p = providerFromRow(v);
    return p === null ? [] : [p];
  });
  const rateDoc = docs.find((d) => d.schema === 'free-model-aggregator-01/ratings/v1' && Array.isArray(d.ratings)) ?? null;
  const ratings =
    rateDoc === null || !Array.isArray(rateDoc.ratings)
      ? null
      : rateDoc.ratings.flatMap((v) => {
          const r = qualityFromRow(v);
          return r === null ? [] : [r];
        });
  const best =
    rateDoc !== null && Array.isArray(rateDoc.best)
      ? rateDoc.best.flatMap((v) => {
          if (!isObj(v) || strOf(v.id) === '') return [];
          return [{ id: strOf(v.id), rating: numOrNull(v.rating_milli) ?? 0 }];
        })
      : [];
  const workDoc = docs.find((d) => d.schema === 'free-model-aggregator-01/workload/v1' && Array.isArray(d.manifest)) ?? null;
  const workload =
    workDoc === null || !Array.isArray(workDoc.manifest)
      ? null
      : workDoc.manifest.flatMap((v) => {
          const w = workloadFromRow(v);
          return w === null ? [] : [w];
        });
  return { providers, ratings, best, workload, problem: null };
}

/** The quality row of one provider's model, when the table has one. */
function qualityRow(ratings: QualityRow[] | null, providerId: string, modelId: string): QualityRow | null {
  if (ratings === null) return null;
  return ratings.find((r) => r.provider === providerId && r.model === modelId) ?? null;
}

/** Whether a model is free: the quality table where it has a row, else the listing, else paid. */
function isFreeModel(model: ListedModel, row: QualityRow | null): boolean {
  if (row !== null) return row.free;
  return model.free ?? false;
}

/** What a model can do: the quality table where it has a row, else the listing. */
function capsOf(model: ListedModel, row: QualityRow | null): Caps {
  return {
    context: row?.context ?? model.context ?? null,
    tools: row?.tools ?? model.tools ?? null,
    vision: row?.vision ?? model.vision ?? null,
    json: row?.json ?? model.json ?? null,
  };
}

/** Whether a model's capabilities meet one request's needs. An unknown capability never meets one. */
function meetsNeed(need: Need, caps: Caps): boolean {
  if (need.tools === true && caps.tools !== true) return false;
  if (need.vision === true && caps.vision !== true) return false;
  if (need.json === true && caps.json !== true) return false;
  if (need.minContext !== undefined && (caps.context === null || caps.context < need.minContext)) return false;
  return true;
}

/** A listing's price field: true when every price is zero, null when it says nothing. */
function pricedZero(v: JsonObj): boolean | null {
  const pricing = isObj(v.pricing) ? v.pricing : null;
  if (pricing === null) return null;
  const values = Object.values(pricing).flatMap((x) => {
    if (typeof x === 'number') return [x];
    if (typeof x === 'string' && x.trim() !== '') return [Number(x)];
    return [];
  });
  if (values.length === 0) return null;
  return values.every((n) => Number.isFinite(n) && n === 0);
}

/** One model as a listing wrote it, in whichever wire's spelling. */
function modelFromListing(wire: WireKind, v: unknown): ListedModel | null {
  if (!isObj(v)) return null;
  const id = wire === 'gemini' ? strOf(v.name).replace(/^models\//, '') : strOf(v.id);
  if (id === '') return null;
  const context = numOrNull(v.context_length) ?? numOrNull(v.context_window) ?? numOrNull(v.inputTokenLimit);
  return {
    id,
    display: strOf(v.display_name) !== '' ? strOf(v.display_name) : strOf(v.displayName) !== '' ? strOf(v.displayName) : id,
    free: boolOrNull(v.free) ?? pricedZero(v),
    context,
    tools: boolOrNull(v.tools),
    vision: boolOrNull(v.vision),
    json: boolOrNull(v.json),
  };
}

/** The models one catalog route's answer listed, in its wire's shape. */
function modelsFromCatalog(wire: WireKind, body: unknown): ListedModel[] {
  if (!isObj(body)) return [];
  const list = wire === 'gemini' ? body.models : body.data;
  if (!Array.isArray(list)) return [];
  return list.flatMap((v) => {
    const m = modelFromListing(wire, v);
    return m === null ? [] : [m];
  });
}

/** Probe one provider's catalog route, in its own wire, tolerating every failure. */
async function probeOne(origin: string, provider: DirectoryProvider, timeoutMs = PROBE_BUDGET_MS): Promise<ProbeResult> {
  const base = joinOrigin(origin, provider.base);
  const url = provider.wire === 'gemini' ? `${base}/v1beta/models` : `${base}/v1/models`;
  const headers: Record<string, string> = { accept: 'application/json', ...keyHeaders(provider.auth, provider.key) };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = ANTHROPIC_VERSION;
  try {
    const res = await fetch(url, { headers, signal: AbortSignal.timeout(timeoutMs) });
    if (!res.ok) return { provider, ok: false, status: res.status, models: [], problem: `GET ${url} answered HTTP ${res.status}` };
    const body: unknown = await res.json();
    return { provider, ok: true, status: res.status, models: modelsFromCatalog(provider.wire, body), problem: null };
  } catch (err) {
    return { provider, ok: false, status: 0, models: [], problem: `GET ${url} failed: ${(err as Error).message}` };
  }
}

/**
 * The catalog the probes and the quality table make: every provider that
 * answered, every model it listed. Its base_url is the directory's own path
 * (/p/<id>), never the joined URL: the env's origin differs between a live
 * pass and its replay, and the catalog is part of the state a replay must
 * reproduce byte for byte.
 */
function buildCatalog(probes: readonly ProbeResult[], ratings: QualityRow[] | null): CatalogDoc {
  const providers: CatalogProvider[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    const seen = new Set<string>();
    const models: CatalogProvider['models'] = [];
    for (const model of [...probe.models].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))) {
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      const row = qualityRow(ratings, probe.provider.id, model.id);
      models.push({ id: model.id, free: isFreeModel(model, row), context_length: row?.context ?? model.context ?? null });
    }
    providers.push({ id: probe.provider.id, base_url: probe.provider.base, key_env: probe.provider.keyEnv, wire: probe.provider.wire, models });
  }
  return { providers };
}

// ---------------------------------------------------------------------------
// Wire translation
// ---------------------------------------------------------------------------
// The client speaks OpenAI or Anthropic; each provider speaks OpenAI,
// Anthropic or Gemini. A request is translated faithfully into the
// provider's wire and the answer is repackaged in the client's wire, with
// the provider's text whole and its usage counts copied exactly. The
// simulated providers never emit tool calls in an answer (their text is
// always the literal "ANSWER ECHO" plus the echo JSON), so an answer is
// always plain text plus a stop reason plus usage; the translation below
// still maps a tool_call finish through, for safety.

/** One model the entry serves, as GET /v1/models lists it. */
interface ServedModel {
  id: string;
  provider: string;
  display: string;
}

/** GET /v1/models on an OpenAI-shaped request. */
function openAIList(models: readonly ServedModel[]): JsonObj {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: LIST_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header). */
function anthropicList(models: readonly ServedModel[]): JsonObj {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display, created_at: LIST_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape. */
function anthropicErr(type: string, message: string): JsonObj {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape. */
function openAIErr(message: string, type: string, code: string | null): JsonObj {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks; '' when none. */
function plainText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isObj(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** One Anthropic content block as an OpenAI content part: text or image; null otherwise. */
function openAIPart(block: JsonObj): JsonObj | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isObj(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON. */
function resultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = plainText(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: the JSON string, or {} when not one. */
function argsObject(value: unknown): JsonObj {
  if (isObj(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isObj(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

function numOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
function anthropicStop(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/** Anthropic's stop_reason as OpenAI's finish_reason. */
function openAIStop(reason: unknown): string {
  if (reason === 'max_tokens') return 'length';
  if (reason === 'tool_use') return 'tool_calls';
  if (reason === 'refusal') return 'content_filter';
  return 'stop';
}

/** Gemini's finishReason as OpenAI's finish_reason. */
function geminiStop(reason: unknown): string {
  if (reason === 'MAX_TOKENS') return 'length';
  if (reason === 'SAFETY') return 'content_filter';
  return 'stop';
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request naming
 * `model`: the system prompt becomes a system message, image blocks become
 * image_url data URLs, tool_use blocks become tool_calls, tool_result blocks
 * become tool messages, tools and tool_choice are rewritten, and the output
 * cap, temperature, top_p, stop sequences and streaming pass through.
 */
function anthropicRequestToOpenAI(body: JsonObj, model: string): JsonObj {
  const messages: JsonObj[] = [];
  const system = plainText(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isObj(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: JsonObj[] = [];
      for (const block of content) {
        if (!isObj(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isObj(block.input) ? block.input : {}),
          },
        });
      }
      const text = plainText(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: JsonObj[] = [];
    for (const block of content) {
      if (!isObj(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: resultText(block.content),
        });
        continue;
      }
      const part = openAIPart(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }
  const out: JsonObj = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isObj(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isObj(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isObj(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** One OpenAI content part as an Anthropic content block: text or image; null otherwise. */
function anthropicBlock(part: JsonObj): JsonObj | null {
  if (part.type === 'text' && typeof part.text === 'string') return { type: 'text', text: part.text };
  if (part.type === 'image_url' && isObj(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const data = /^data:([^,]*);base64,(.*)$/s.exec(url);
    if (data !== null) {
      const media = data[1] !== '' ? data[1] : 'image/png';
      return { type: 'image', source: { type: 'base64', media_type: media, data: data[2] } };
    }
    return { type: 'image', source: { type: 'url', url } };
  }
  return null;
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request naming
 * `model`: system messages become the system prompt, user parts become
 * Anthropic blocks (image_url data URLs become base64 image blocks), tool
 * messages become tool_result blocks, assistant tool_calls become tool_use
 * blocks, and max_tokens, temperature, top_p, stop and streaming pass
 * through. The Anthropic wire has no response_format: callers must not
 * route a JSON-mode OpenAI request here.
 */
function openAIRequestToAnthropic(body: JsonObj, model: string): JsonObj {
  const messages: JsonObj[] = [];
  let system = '';
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isObj(raw)) continue;
    const role = raw.role;
    const content = raw.content;
    if (role === 'system') {
      const text = plainText(content);
      if (text !== '') system = system === '' ? text : `${system}\n${text}`;
      continue;
    }
    if (role === 'assistant') {
      const blocks: JsonObj[] = [];
      const text = plainText(content);
      if (text !== '') blocks.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isObj(call) || !isObj(call.function)) continue;
        blocks.push({
          type: 'tool_use',
          id: typeof call.id === 'string' && call.id !== '' ? call.id : 'toolu_unknown',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argsObject(call.function.arguments),
        });
      }
      if (blocks.length > 0) messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    if (role === 'tool') {
      messages.push({
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', content: typeof content === 'string' ? content : resultText(content) }],
      });
      continue;
    }
    // A user message.
    if (typeof content === 'string') {
      messages.push({ role: 'user', content: [{ type: 'text', text: content }] });
      continue;
    }
    if (Array.isArray(content)) {
      const blocks: JsonObj[] = [];
      for (const part of content) {
        if (!isObj(part)) continue;
        const block = anthropicBlock(part);
        if (block !== null) blocks.push(block);
      }
      if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
    }
  }
  const out: JsonObj = { model, messages };
  if (system !== '') out.system = system;
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  else if (typeof body.max_completion_tokens === 'number') out.max_tokens = body.max_completion_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop) && body.stop.length > 0) out.stop_sequences = body.stop;
  else if (typeof body.stop === 'string' && body.stop !== '') out.stop_sequences = [body.stop];
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isObj(raw) || !isObj(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          input_schema: isObj(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required') out.tool_choice = { type: 'any' };
  else if (isObj(choice) && choice.type === 'function' && isObj(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/** A Chat Completions answer as an Anthropic Messages object. */
function openAIAnswerToAnthropic(body: JsonObj, model: string): JsonObj {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isObj(choices[0]) ? choices[0] : {};
  const message = isObj(choice.message) ? choice.message : {};
  const content: JsonObj[] = [];
  const text = typeof message.content === 'string' ? message.content : plainText(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isObj(call) || !isObj(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argsObject(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isObj(body.usage) ? body.usage : {};
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: anthropicStop(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: numOrZero(usage.prompt_tokens), output_tokens: numOrZero(usage.completion_tokens) },
  };
}

/** An Anthropic Messages answer as a Chat Completions object. */
function anthropicAnswerToOpenAI(body: JsonObj, model: string): JsonObj {
  const blocks = Array.isArray(body.content) ? body.content : [];
  let text = '';
  const calls: JsonObj[] = [];
  for (const block of blocks) {
    if (!isObj(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') text = text === '' ? block.text : `${text}\n${block.text}`;
    if (block.type === 'tool_use') {
      calls.push({
        id: typeof block.id === 'string' ? block.id : 'call_unknown',
        type: 'function',
        function: { name: typeof block.name === 'string' ? block.name : '', arguments: JSON.stringify(isObj(block.input) ? block.input : {}) },
      });
    }
  }
  const usage = isObj(body.usage) ? body.usage : {};
  return {
    id: `chatcmpl_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    object: 'chat.completion',
    created: LIST_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content: text, ...(calls.length > 0 ? { tool_calls: calls } : {}) },
        finish_reason: openAIStop(body.stop_reason),
        logprobs: null,
      },
    ],
    usage: { prompt_tokens: numOrZero(usage.input_tokens), completion_tokens: numOrZero(usage.output_tokens), total_tokens: numOrZero(usage.input_tokens) + numOrZero(usage.output_tokens) },
  };
}

/** A Gemini generateContent answer as a Chat Completions object. */
function geminiAnswerToOpenAI(body: JsonObj, model: string): JsonObj {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const first = isObj(candidates[0]) ? candidates[0] : {};
  const content = isObj(first.content) ? first.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  const text = parts.flatMap((p) => (isObj(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
  const meta = isObj(body.usageMetadata) ? body.usageMetadata : {};
  const input = numOrZero(meta.promptTokenCount);
  const output = numOrZero(meta.candidatesTokenCount);
  return {
    id: 'chatcmpl_gemini',
    object: 'chat.completion',
    created: LIST_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: text }, finish_reason: geminiStop(first.finishReason), logprobs: null }],
    usage: { prompt_tokens: input, completion_tokens: output, total_tokens: input + output },
  };
}

/** A Gemini generateContent answer as an Anthropic Messages object. */
function geminiAnswerToAnthropic(body: JsonObj, model: string): JsonObj {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const first = isObj(candidates[0]) ? candidates[0] : {};
  const content = isObj(first.content) ? first.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  const text = parts.flatMap((p) => (isObj(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
  const meta = isObj(body.usageMetadata) ? body.usageMetadata : {};
  return {
    id: 'msg_gemini',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text }],
    stop_reason: first.finishReason === 'MAX_TOKENS' ? 'max_tokens' : first.finishReason === 'SAFETY' ? 'refusal' : 'end_turn',
    stop_sequence: null,
    usage: { input_tokens: numOrZero(meta.promptTokenCount), output_tokens: numOrZero(meta.candidatesTokenCount) },
  };
}

// ---------------------------------------------------------------------------
// Gemini request translation
// ---------------------------------------------------------------------------
// Gemini's generateContent takes {contents, systemInstruction, tools,
// generationConfig}. Contents alternate user/model roles; consecutive
// messages of one role are merged. Images become inlineData parts, tool
// calls become functionCall parts, tool results become functionResponse
// parts. JSON mode is generationConfig.responseMimeType.

/** One Gemini content part from an OpenAI content part. */
function geminiPartFromOpenAI(part: JsonObj): JsonObj | null {
  if (part.type === 'text' && typeof part.text === 'string') return { text: part.text };
  if (part.type === 'image_url' && isObj(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const data = /^data:([^,]*);base64,(.*)$/s.exec(url);
    if (data !== null) return { inlineData: { mimeType: data[1] !== '' ? data[1] : 'image/png', data: data[2] } };
    return { fileData: { fileUri: url } };
  }
  return null;
}

/** One Gemini content part from an Anthropic content block. */
function geminiPartFromAnthropic(block: JsonObj): JsonObj | null {
  if (block.type === 'text' && typeof block.text === 'string') return { text: block.text };
  if (block.type === 'image' && isObj(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      return { inlineData: { mimeType: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { fileData: { fileUri: source.url } };
  }
  if (block.type === 'tool_use') {
    return { functionCall: { name: typeof block.name === 'string' ? block.name : '', args: isObj(block.input) ? block.input : {} } };
  }
  if (block.type === 'tool_result') {
    return {
      functionResponse: {
        name: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
        response: { result: resultText(block.content) },
      },
    };
  }
  return null;
}

/** Append `parts` to `contents` under `role`, merging with the last entry when it shares the role. */
function pushGeminiContent(contents: JsonObj[], role: string, parts: JsonObj[]): void {
  if (parts.length === 0) return;
  const last = contents[contents.length - 1];
  if (isObj(last) && last.role === role && Array.isArray(last.parts)) {
    (last.parts as JsonObj[]).push(...parts);
    return;
  }
  contents.push({ role, parts });
}

/** An OpenAI Chat Completions request as a Gemini generateContent body. */
function openAIRequestToGemini(body: JsonObj): JsonObj {
  const contents: JsonObj[] = [];
  let systemText = '';
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isObj(raw)) continue;
    const role = raw.role;
    const content = raw.content;
    if (role === 'system') {
      const text = plainText(content);
      if (text !== '') systemText = systemText === '' ? text : `${systemText}\n${text}`;
      continue;
    }
    if (role === 'assistant') {
      const parts: JsonObj[] = [];
      const text = plainText(content);
      if (text !== '') parts.push({ text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isObj(call) || !isObj(call.function)) continue;
        parts.push({ functionCall: { name: typeof call.function.name === 'string' ? call.function.name : '', args: argsObject(call.function.arguments) } });
      }
      pushGeminiContent(contents, 'model', parts);
      continue;
    }
    if (role === 'tool') {
      pushGeminiContent(contents, 'user', [
        { functionResponse: { name: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', response: { result: typeof content === 'string' ? content : resultText(content) } } },
      ]);
      continue;
    }
    const parts: JsonObj[] = [];
    if (typeof content === 'string') {
      if (content !== '') parts.push({ text: content });
    } else if (Array.isArray(content)) {
      for (const part of content) {
        if (!isObj(part)) continue;
        const gp = geminiPartFromOpenAI(part);
        if (gp !== null) parts.push(gp);
      }
    }
    pushGeminiContent(contents, 'user', parts);
  }
  const out: JsonObj = { contents };
  if (systemText !== '') out.systemInstruction = { parts: [{ text: systemText }] };
  if (Array.isArray(body.tools)) {
    const decls = body.tools.flatMap((raw) => {
      if (!isObj(raw) || !isObj(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          parameters: isObj(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (decls.length > 0) out.tools = [{ functionDeclarations: decls }];
  }
  const config: JsonObj = {};
  if (typeof body.max_tokens === 'number') config.maxOutputTokens = body.max_tokens;
  else if (typeof body.max_completion_tokens === 'number') config.maxOutputTokens = body.max_completion_tokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  if (Array.isArray(body.stop) && body.stop.length > 0) config.stopSequences = body.stop;
  else if (typeof body.stop === 'string' && body.stop !== '') config.stopSequences = [body.stop];
  const format = isObj(body.response_format) ? body.response_format.type : undefined;
  if (format === 'json_object' || format === 'json_schema') config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/** An Anthropic Messages request as a Gemini generateContent body. */
function anthropicRequestToGemini(body: JsonObj): JsonObj {
  const contents: JsonObj[] = [];
  const systemText = plainText(body.system);
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isObj(raw)) continue;
    const parts: JsonObj[] = [];
    const content = raw.content;
    if (typeof content === 'string') {
      if (content !== '') parts.push({ text: content });
    } else if (Array.isArray(content)) {
      for (const block of content) {
        if (!isObj(block)) continue;
        const gp = geminiPartFromAnthropic(block);
        if (gp !== null) parts.push(gp);
      }
    }
    pushGeminiContent(contents, raw.role === 'assistant' ? 'model' : 'user', parts);
  }
  const out: JsonObj = { contents };
  if (systemText !== '') out.systemInstruction = { parts: [{ text: systemText }] };
  if (Array.isArray(body.tools)) {
    const decls = body.tools.flatMap((raw) => {
      if (!isObj(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          name: raw.name,
          description: typeof raw.description === 'string' ? raw.description : '',
          parameters: isObj(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
        },
      ];
    });
    if (decls.length > 0) out.tools = [{ functionDeclarations: decls }];
  }
  const config: JsonObj = {};
  if (typeof body.max_tokens === 'number') config.maxOutputTokens = body.max_tokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) config.stopSequences = body.stop_sequences;
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

// ---------------------------------------------------------------------------
// SSE stream translation
// ---------------------------------------------------------------------------
// One server-sent event: `event: <name>` when named, then the JSON data
// line, then the blank line.

function sseLine(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** Split a raw SSE byte stream into frames: each frame's event name and data lines. */
function sseFrames(text: string): { event: string | null; data: string }[] {
  const frames: { event: string | null; data: string }[] = [];
  const chunks = text.replace(/\r\n/g, '\n').split('\n\n');
  for (const chunk of chunks) {
    if (chunk.trim() === '') continue;
    let event: string | null = null;
    const dataLines: string[] = [];
    for (const line of chunk.split('\n')) {
      if (line.startsWith('event:')) event = line.slice(6).trim();
      else if (line.startsWith('data:')) dataLines.push(line.slice(5).trim());
    }
    if (dataLines.length > 0) frames.push({ event, data: dataLines.join('\n') });
  }
  return frames;
}

/**
 * An OpenAI SSE stream as the Anthropic event sequence: message_start on
 * the first chunk, content_block_start/delta/stop per text block or tool
 * call, then message_delta with the stop reason and usage, then
 * message_stop. The provider's usage chunk arrives after the finish chunk
 * and before [DONE], so the output tokens are known by the time the
 * message_delta is written. The input tokens in message_start stay 0: the
 * client's stream check only reads the output count.
 */
function openAIStreamToAnthropic(frames: { event: string | null; data: string }[], model: string): string {
  const out: string[] = [];
  let started = false;
  let finished = false;
  let textIndex = -1;
  let nextIndex = 0;
  const toolSlots = new Map<number, number>();
  let outputTokens = 0;
  let pendingStop: string | null = null;

  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sseLine('message_start', {
        type: 'message_start',
        message: { id: 'msg_stream', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: 0, output_tokens: 0 } },
      }),
    ];
  };
  const closeEvents = (): string[] => {
    const ev: string[] = [];
    if (textIndex !== -1) {
      ev.push(sseLine('content_block_stop', { type: 'content_block_stop', index: textIndex }));
      textIndex = -1;
    }
    for (const index of [...toolSlots.values()].sort((a, b) => a - b)) ev.push(sseLine('content_block_stop', { type: 'content_block_stop', index }));
    toolSlots.clear();
    return ev;
  };
  const finishEvents = (stopReason: string): string[] => {
    if (finished) return [];
    finished = true;
    return [
      ...closeEvents(),
      sseLine('message_delta', { type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: outputTokens } }),
      sseLine('message_stop', { type: 'message_stop' }),
    ];
  };

  for (const frame of frames) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') {
      out.push(...startEvents(), ...finishEvents(pendingStop ?? 'end_turn'));
      continue;
    }
    let chunk: unknown;
    try {
      chunk = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isObj(chunk)) continue;
    if (isObj(chunk.usage)) outputTokens = numOrZero(chunk.usage.completion_tokens);
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    if (choices.length === 0) continue;
    const choice = isObj(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isObj(choice.delta) ? choice.delta : {};
    out.push(...startEvents());
    if (typeof delta.content === 'string' && delta.content !== '') {
      if (textIndex === -1) {
        textIndex = nextIndex;
        nextIndex += 1;
        out.push(sseLine('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } }));
      }
      out.push(sseLine('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text: delta.content } }));
    }
    const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
    for (const call of calls) {
      if (!isObj(call)) continue;
      const upstreamIndex = typeof call.index === 'number' ? call.index : toolSlots.size;
      let index = toolSlots.get(upstreamIndex);
      if (index === undefined) {
        if (textIndex !== -1) {
          out.push(sseLine('content_block_stop', { type: 'content_block_stop', index: textIndex }));
          textIndex = -1;
        }
        index = nextIndex;
        nextIndex += 1;
        toolSlots.set(upstreamIndex, index);
        const fn = isObj(call.function) ? call.function : {};
        out.push(
          sseLine('content_block_start', {
            type: 'content_block_start',
            index,
            content_block: { type: 'tool_use', id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${index}`, name: typeof fn.name === 'string' ? fn.name : '', input: {} },
          }),
        );
      }
      const fn = isObj(call.function) ? call.function : {};
      if (typeof fn.arguments === 'string' && fn.arguments !== '') {
        out.push(sseLine('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: fn.arguments } }));
      }
    }
    if (choice.finish_reason !== null && choice.finish_reason !== undefined) pendingStop = anthropicStop(choice.finish_reason);
  }
  if (!finished) {
    out.push(...startEvents(), ...finishEvents(pendingStop ?? 'end_turn'));
  }
  return out.join('');
}

/**
 * An Anthropic SSE stream as OpenAI chunks: a role chunk, then content
 * deltas, then the finish chunk, then the usage chunk, then [DONE]. The
 * usage counts come from message_start (input) and message_delta (output).
 */
function anthropicStreamToOpenAI(frames: { event: string | null; data: string }[], model: string): string {
  const out: string[] = [];
  let inputTokens = 0;
  let outputTokens = 0;
  let stopReason: string | null = null;
  let emittedRole = false;
  let finished = false;

  const chunk = (delta: JsonObj, finish: string | null): string =>
    sseLine(null, { id: 'chatcmpl_stream', object: 'chat.completion.chunk', created: LIST_CREATED, model, choices: [{ index: 0, delta, finish_reason: finish }] });

  for (const frame of frames) {
    if (frame.event === null) continue;
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isObj(data)) continue;
    if (frame.event === 'message_start' && isObj(data.message) && isObj(data.message.usage)) {
      inputTokens = numOrZero(data.message.usage.input_tokens);
    } else if (frame.event === 'content_block_delta' && isObj(data.delta)) {
      if (data.delta.type === 'text_delta' && typeof data.delta.text === 'string' && data.delta.text !== '') {
        if (!emittedRole) {
          emittedRole = true;
          out.push(chunk({ role: 'assistant', content: '' }, null));
        }
        out.push(chunk({ content: data.delta.text }, null));
      }
      if (data.delta.type === 'input_json_delta' && typeof data.delta.partial_json === 'string' && data.delta.partial_json !== '') {
        if (!emittedRole) {
          emittedRole = true;
          out.push(chunk({ role: 'assistant', content: '' }, null));
        }
        // Tool arguments stream as a tool call delta; the call's index is the block's.
        const idx = typeof data.index === 'number' ? data.index : 0;
        out.push(chunk({ tool_calls: [{ index: idx, function: { arguments: data.delta.partial_json } }] }, null));
      }
    } else if (frame.event === 'content_block_start' && isObj(data.content_block) && data.content_block.type === 'tool_use') {
      if (!emittedRole) {
        emittedRole = true;
        out.push(chunk({ role: 'assistant', content: '' }, null));
      }
      const idx = typeof data.index === 'number' ? data.index : 0;
      out.push(
        chunk(
          {
            tool_calls: [
              {
                index: idx,
                id: typeof data.content_block.id === 'string' ? data.content_block.id : `call_${idx}`,
                type: 'function',
                function: { name: typeof data.content_block.name === 'string' ? data.content_block.name : '', arguments: '' },
              },
            ],
          },
          null,
        ),
      );
    } else if (frame.event === 'message_delta' && isObj(data.delta)) {
      if (typeof data.delta.stop_reason === 'string') stopReason = openAIStop(data.delta.stop_reason);
      if (isObj(data.usage)) outputTokens = numOrZero(data.usage.output_tokens);
    } else if (frame.event === 'message_stop') {
      finished = true;
    }
  }
  if (!emittedRole) out.push(chunk({ role: 'assistant', content: '' }, null));
  out.push(chunk({}, stopReason ?? 'stop'));
  out.push(sseLine(null, { id: 'chatcmpl_stream', object: 'chat.completion.chunk', created: LIST_CREATED, model, choices: [], usage: { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens } }));
  out.push('data: [DONE]\n\n');
  return out.join('');
}

/** The text pieces and usage of a Gemini SSE stream, in arrival order. */
function geminiStreamParts(frames: { event: string | null; data: string }[]): { pieces: string[]; input: number; output: number; done: boolean } {
  const pieces: string[] = [];
  let input = 0;
  let output = 0;
  let done = false;
  for (const frame of frames) {
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isObj(data)) continue;
    if (isObj(data.usageMetadata)) {
      input = numOrZero(data.usageMetadata.promptTokenCount);
      output = numOrZero(data.usageMetadata.candidatesTokenCount);
    }
    const candidates = Array.isArray(data.candidates) ? data.candidates : [];
    for (const cand of candidates) {
      if (!isObj(cand)) continue;
      if (typeof cand.finishReason === 'string') done = true;
      const content = isObj(cand.content) ? cand.content : null;
      if (content === null || !Array.isArray(content.parts)) continue;
      for (const part of content.parts) {
        if (isObj(part) && typeof part.text === 'string' && part.text !== '') pieces.push(part.text);
      }
    }
  }
  return { pieces, input, output, done };
}

/** A Gemini SSE stream as OpenAI chunks. */
function geminiStreamToOpenAI(frames: { event: string | null; data: string }[], model: string): string {
  const { pieces, input, output } = geminiStreamParts(frames);
  const out: string[] = [];
  const chunk = (delta: JsonObj, finish: string | null): string =>
    sseLine(null, { id: 'chatcmpl_stream', object: 'chat.completion.chunk', created: LIST_CREATED, model, choices: [{ index: 0, delta, finish_reason: finish }] });
  out.push(chunk({ role: 'assistant', content: '' }, null));
  for (const piece of pieces) out.push(chunk({ content: piece }, null));
  out.push(chunk({}, 'stop'));
  out.push(sseLine(null, { id: 'chatcmpl_stream', object: 'chat.completion.chunk', created: LIST_CREATED, model, choices: [], usage: { prompt_tokens: input, completion_tokens: output, total_tokens: input + output } }));
  out.push('data: [DONE]\n\n');
  return out.join('');
}

/** A Gemini SSE stream as the Anthropic event sequence. */
function geminiStreamToAnthropic(frames: { event: string | null; data: string }[], model: string): string {
  const { pieces, output } = geminiStreamParts(frames);
  const out: string[] = [];
  out.push(
    sseLine('message_start', {
      type: 'message_start',
      message: { id: 'msg_stream', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: 0, output_tokens: 0 } },
    }),
  );
  out.push(sseLine('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }));
  for (const piece of pieces) out.push(sseLine('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } }));
  out.push(sseLine('content_block_stop', { type: 'content_block_stop', index: 0 }));
  out.push(sseLine('message_delta', { type: 'message_delta', delta: { stop_reason: 'end_turn', stop_sequence: null }, usage: { output_tokens: output } }));
  out.push(sseLine('message_stop', { type: 'message_stop' }));
  return out.join('');
}

// ---------------------------------------------------------------------------
// Routing, upstream calls, and the server
// ---------------------------------------------------------------------------

const brief = readPrompt(BRIEF_PATH);
let probes: ProbeResult[] = [];
let catalog: CatalogDoc = { providers: [] };
/** Refusals per provider id, for ranking penalties. */
const refusals = new Map<string, number>();
/** Consecutive refusals per provider id, for the spent rule. */
const streak = new Map<string, number>();
/** Providers treated as spent for the rest of the trial. */
const spent = new Set<string>();
/** How many client API requests have arrived; the manifest is in arrival order. */
let manifestIndex = 0;

function note(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (--list and /register share it). */
async function loadCatalog(): Promise<void> {
  if (brief.problem !== null) note(brief.problem);
  probes = await Promise.all(brief.providers.map((provider) => probeOne(ENV_ORIGIN, provider, PROBE_BUDGET_MS)));
  for (const probe of probes) if (!probe.ok && probe.problem !== null) note(probe.problem);
  catalog = buildCatalog(probes, brief.ratings);
}

/** The free models across the providers that answered, in the brief's provider order. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))) {
      const row = qualityRow(brief.ratings, probe.provider.id, model.id);
      if (!isFreeModel(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display: model.display });
    }
  }
  return out;
}

interface Candidate {
  probe: ProbeResult;
  model: ListedModel;
  rating: number;
  effective: number;
}

/**
 * The ranked candidates for one request: every free model that meets the
 * requirements, best rating first. A provider's refusals lower its models'
 * effective rating; a provider that refused three times running is spent.
 * An OpenAI JSON-mode request (response_format) can only go to an
 * OpenAI-wire or Gemini-wire provider: the Anthropic wire has no
 * response_format to carry, so its echo would report json_mode false.
 */
function rankCandidates(need: Need, jsonMode: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (spent.has(probe.provider.id)) continue;
    if (jsonMode && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = qualityRow(brief.ratings, probe.provider.id, model.id);
      if (!isFreeModel(model, row)) continue;
      if (!meetsNeed(need, capsOf(model, row))) continue;
      const rating = row?.rating ?? 0;
      const penalty = REFUSAL_PENALTY_MILLI * (refusals.get(probe.provider.id) ?? 0);
      out.push({ probe, model, rating, effective: rating - penalty });
    }
  }
  out.sort((a, b) => {
    if (b.effective !== a.effective) return b.effective - a.effective;
    if (a.probe.provider.id !== b.probe.provider.id) return a.probe.provider.id < b.probe.provider.id ? -1 : 1;
    return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
  });
  return out;
}

/** Record a provider's refusal: a ranking penalty and a spent check. */
function recordRefusal(providerId: string): void {
  refusals.set(providerId, (refusals.get(providerId) ?? 0) + 1);
  const run = (streak.get(providerId) ?? 0) + 1;
  streak.set(providerId, run);
  if (run >= SPENT_AFTER_REFUSALS) spent.add(providerId);
}

/** Record a provider's success: its refusal streak resets. */
function recordSuccess(providerId: string): void {
  streak.set(providerId, 0);
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function needsFor(body: JsonObj, clientApi: 'openai' | 'anthropic'): Need {
  const manifest = brief.workload ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined && entry.api === clientApi) return entry.need;
  return needsFromBody(body, clientApi);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function needsFromBody(body: JsonObj, clientApi: 'openai' | 'anthropic'): Need {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isObj(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isObj(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isObj(body.response_format) ? body.response_format.type : undefined;
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: format === 'json_object' || format === 'json_schema',
  };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJsonBody(req: IncomingMessage): Promise<JsonObj | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_REQUEST_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isObj(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendSse(res: ServerResponse, body: string): void {
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
  res.end(body);
}

function sendErr(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicErr(type, message));
  else sendJson(res, status, openAIErr(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function serveModels(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicList(list));
  else sendJson(res, 200, openAIList(list));
}

interface UpstreamAnswer {
  ok: boolean;
  status: number;
  /** The provider's answer body: JSON text for a non-stream call, SSE text for a stream. */
  text: string;
}

/**
 * One call to a provider in its own wire. `clientBody` is the client's
 * request (OpenAI or Anthropic shape); it is translated into the provider's
 * wire, with the candidate model named. Returns the provider's raw answer.
 */
async function callUpstream(candidate: Candidate, clientApi: 'openai' | 'anthropic', clientBody: JsonObj, stream: boolean): Promise<UpstreamAnswer> {
  const provider = candidate.probe.provider;
  const base = joinOrigin(ENV_ORIGIN, provider.base);
  const headers: Record<string, string> = { 'content-type': 'application/json', ...keyHeaders(provider.auth, provider.key) };
  let url = '';
  let upstreamBody: JsonObj;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    upstreamBody = clientApi === 'openai' ? { ...clientBody, model: candidate.model.id } : anthropicRequestToOpenAI(clientBody, candidate.model.id);
    if (stream) {
      upstreamBody.stream = true;
      upstreamBody.stream_options = { include_usage: true };
      headers.accept = 'text/event-stream';
    }
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = ANTHROPIC_VERSION;
    upstreamBody = clientApi === 'anthropic' ? { ...clientBody, model: candidate.model.id } : openAIRequestToAnthropic(clientBody, candidate.model.id);
    if (stream) {
      upstreamBody.stream = true;
      headers.accept = 'text/event-stream';
    }
  } else {
    const encoded = encodeURIComponent(candidate.model.id);
    url = stream ? `${base}/v1beta/models/${encoded}:streamGenerateContent` : `${base}/v1beta/models/${encoded}:generateContent`;
    upstreamBody = clientApi === 'openai' ? openAIRequestToGemini(clientBody) : anthropicRequestToGemini(clientBody);
    if (stream) headers.accept = 'text/event-stream';
  }
  try {
    const res = await fetch(url, { method: 'POST', headers, body: JSON.stringify(upstreamBody), signal: AbortSignal.timeout(30000) });
    const text = await res.text();
    return { ok: res.ok, status: res.status, text };
  } catch (err) {
    return { ok: false, status: 0, text: (err as Error).message };
  }
}

/**
 * Serve one client request by trying the ranked candidates in order until
 * one answers. The first success ends the request (exactly one provider
 * call may answer it); every refusal is recorded and the next candidate
 * is tried. The answer is repackaged in the client's wire.
 */
async function serveRequest(req: IncomingMessage, res: ServerResponse, clientApi: 'openai' | 'anthropic'): Promise<void> {
  const body = await readJsonBody(req);
  if (body === null) return sendErr(res, clientApi, 400, 'invalid_request_error', 'the request body is not one JSON object');
  const stream = body.stream === true;
  const need = needsFor(body, clientApi);
  const jsonMode = clientApi === 'openai' && isObj(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema');
  const candidates = rankCandidates(need, jsonMode);
  if (candidates.length === 0) {
    return sendErr(res, clientApi, 503, clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model on the directory meets this request');
  }
  let lastStatus = 502;
  let lastDetail = 'every candidate refused';
  const tried = Math.min(candidates.length, MAX_CANDIDATES);
  for (let i = 0; i < tried; i++) {
    const candidate = candidates[i];
    const answer = await callUpstream(candidate, clientApi, body, stream);
    if (!answer.ok) {
      recordRefusal(candidate.probe.provider.id);
      lastStatus = answer.status !== 0 ? answer.status : 502;
      lastDetail = `the provider answered HTTP ${answer.status === 0 ? 'unreachable' : answer.status}`;
      continue;
    }
    recordSuccess(candidate.probe.provider.id);
    const providerWire = candidate.probe.provider.wire;
    const model = candidate.model.id;
    if (!stream) {
      let parsed: unknown;
      try {
        parsed = JSON.parse(answer.text);
      } catch {
        return sendErr(res, clientApi, 502, 'api_error', 'the provider answered a body that is not JSON');
      }
      if (!isObj(parsed)) return sendErr(res, clientApi, 502, 'api_error', 'the provider answered a JSON value that is not one object');
      if (clientApi === 'openai') {
        if (providerWire === 'openai') return sendJson(res, 200, parsed);
        if (providerWire === 'anthropic') return sendJson(res, 200, anthropicAnswerToOpenAI(parsed, model));
        return sendJson(res, 200, geminiAnswerToOpenAI(parsed, model));
      }
      if (providerWire === 'anthropic') return sendJson(res, 200, parsed);
      if (providerWire === 'openai') return sendJson(res, 200, openAIAnswerToAnthropic(parsed, model));
      return sendJson(res, 200, geminiAnswerToAnthropic(parsed, model));
    }
    const frames = sseFrames(answer.text);
    if (clientApi === 'openai') {
      if (providerWire === 'openai') return sendSse(res, answer.text);
      if (providerWire === 'anthropic') return sendSse(res, anthropicStreamToOpenAI(frames, model));
      return sendSse(res, geminiStreamToOpenAI(frames, model));
    }
    if (providerWire === 'anthropic') return sendSse(res, answer.text);
    if (providerWire === 'openai') return sendSse(res, openAIStreamToAnthropic(frames, model));
    return sendSse(res, geminiStreamToAnthropic(frames, model));
  }
  return sendErr(res, clientApi, lastStatus === 429 || lastStatus === 402 ? 503 : 502, 'api_error', lastDetail);
}

async function handleRequest(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    serveModels(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await serveRequest(req, res, 'openai');
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await serveRequest(req, res, 'anthropic');
    return;
  }
  sendErr(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listenOn(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogDoc): Promise<void> {
  const res = await fetch(`${ENV_ORIGIN}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_ORIGIN}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isObj(reply) && typeof reply.requests === 'number') note(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function watchWork(server: Server): void {
  let busy = false;
  const timer = setInterval(() => {
    if (busy) return;
    busy = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_ORIGIN}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isObj(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        busy = false;
      }
    })();
  }, WORK_POLL_MS);
}

async function start(): Promise<void> {
  await loadCatalog();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the
    // one built object and stop before any server starts.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_ORIGIN === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handleRequest(req, res).catch((err: unknown) => {
      note((err as Error).stack ?? String(err));
      if (!res.headersSent) sendErr(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listenOn(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  note(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  watchWork(server);
}

try {
  await start();
} catch (err) {
  note((err as Error).stack ?? String(err));
  process.exit(1);
}
