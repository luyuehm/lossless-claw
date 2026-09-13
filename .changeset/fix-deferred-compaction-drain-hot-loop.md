---
"@martian-engineering/lossless-claw": patch
---

Fix an unbounded `setImmediate` hot loop in the deferred-compaction debt drain that could pin the host gateway event loop at ~100% CPU.

`drainDeferredCompactionDebtIfIdle` re-scheduled itself immediately whenever the conversation still reported pending debt after a pass. A debt that can never converge — the common `compacted but still over target` case — turned that into a non-terminating `setImmediate` self-reschedule that starved the main agent and made the gateway unresponsive.

The drain now tracks consecutive no-progress passes per session queue and backs off exponentially (0ms → 50ms → … → 1600ms), stopping entirely after 8 consecutive stalls and deferring convergence to the next user turn or host `maintain()`. Progressful passes reset the counter and keep the immediate path.
