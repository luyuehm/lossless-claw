---
"@martian-engineering/lossless-claw": minor
---

Add a `doctor orphaned` self-healing command that detects and removes orphaned context-engine turn advancements left in OpenClaw's per-agent `context_engine_turn_outbox` table.

A turn advancement normally flows `admitted → accepted → ready` and is deleted once the context engine commits the turn. When the gateway is interrupted mid-turn — or a session is tombstoned during restart recovery — a row can be stranded in `admitted` (still seen as pending by the host drain, pinning the event loop at ~100% CPU and starving the main agent) or `blocked` (terminal, never purged by the core). Neither is reachable from the plugin's `commitTurn`/`maintain` surface, which only manages its own `lcm.db`.

The new command scans every agent database under the OpenClaw state directory, reports stranded rows read-only via `doctor orphaned`, and deletes them via `doctor orphaned apply` after backing up each affected database with `VACUUM INTO`. It is intentionally user-triggered and never modifies host-managed tables automatically.
