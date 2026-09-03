---
"@martian-engineering/lossless-claw": patch
---

lcm_expand_query: tolerate delegated child replies that wrap the JSON contract in prose. The parser now recovers the first complete, balanced JSON object from free-form text (string-literal aware), so a child that prefixes or suffixes thinking text no longer fails with `DELEGATED_EXPANSION_REPLY_INVALID`. The delegated task prompt also now forbids markdown fences and explanatory text outright.
