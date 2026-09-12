---
"@martian-engineering/lossless-claw": patch
---

Add `durableAdvancement` configuration toggle (default `true`) to allow suppressing `turnAdvancementIdempotency: "atomic-idempotent-v1"` in `engine.info`.

When set to `false` via plugin config or `LCM_DURABLE_ADVANCEMENT=false`, OpenClaw's host durable-advancement pipeline is cleanly bypassed, preventing unrecoverable `state: "admitted"` orphaned intents and associated ~100% CPU drain loops upon gateway restart or crash. All context summarization, assembly, compaction, and recall capabilities continue unimpaired via standard `afterTurn` and `assemble` lifecycles.
