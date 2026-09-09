---
"@martian-engineering/lossless-claw": patch
---

Fix a `setImmediate` hot loop that could pin the host gateway's event loop at ~100% CPU and starve the main agent. The idle pending-summary drain re-scheduled itself immediately whenever the compaction coordinator reported `pending: true` with a non-converging reason (e.g. a rebounded or stale frontier that keeps yielding "pending summary work remains"). It now backs off exponentially (0ms → 50ms → … → 1600ms) across consecutive no-progress drains and stops after 8 consecutive stalls, deferring convergence to the next user turn or host `maintain()` instead of spinning forever.
