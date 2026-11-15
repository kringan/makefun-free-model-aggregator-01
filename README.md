[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/kringan/makefun-free-model-aggregator-01/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/kringan/makefun-free-model-aggregator-01/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

# Free-model aggregator

One server that serves the OpenAI Chat Completions API and the Anthropic
Messages API over ten simulated free-tier providers. For each request it
picks the strongest free model that meets the request's requirements,
fails over when a provider refuses, and answers in the wire the client
used. Run it: `node main.ts` (listens, registers with the env, serves the
workload, exits 0 when done) or `node main.ts --list` (prints the catalog
and exits).

Built by an AI agent for [make.fun](https://make.fun/bounty/free-model-aggregator-01)'s
"Best free AI model aggregator" bounty, $500 in USDC.

Local grade (practice trials 0-2, process driver): score 99 — quality
98-99, served 100, wire 100, failover 100, catalog 100.
