# Prompts

`router_v3.md` is the team's router prompt (version v3, written 2026-11-11), copied unchanged from the team repo
`Salikawtar/northstar-agent`, path `prompts/router/v3.md`, `origin/main` commit `9b118f8`. SHA-256 of the copied
file: `3d587ac96476192a40cd9335d89fab19d583e000ba355169f258c199eb312468`.

It is the instruction given to the prompted baselines (Claude Haiku and Claude Sonnet) and the source of the
request-type and escalation wording given to Jev, so all three read the same rules.

It was tuned by the team while looking at the validation chats (30 of our 160 pool chats), so scores of the
prompted models on the pool are slightly optimistic. The 40 test chats are the clean comparison.
