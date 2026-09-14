#!/usr/bin/env bash
# lossless-claw v1.0.0 — air-gap / zero-service-dependency evidence script
# Reproduces the audit evidence for docs/security/*.md on a clean checkout.
set -euo pipefail

REPO="${1:-.}"
cd "$REPO"

echo "==> [1/5] Baseline version + git state"
grep '"version"' package.json
git rev-parse --short HEAD

echo ""
echo "==> [2/5] External import scan (src + bundles)"
echo "--- non-node bare imports in src ---"
grep -rhoE "from ['\"][a-z@][^'\"]*['\"]" src index.ts cli.ts \
  | grep -vE "(node:|\./|\.\./)" | sort -u || true

echo "--- network/service imports in built bundles ---"
grep -noE "node:(http|https|net|dns|http2|tls|child_process)" dist/*.js | head || true

echo ""
echo "==> [3/5] Runtime I/O scan (no external call points)"
echo "--- child_process/spawn ---"
grep -rnE "\b(child_process|spawn|execFile|popen)\b" src index.ts cli.ts \
  | grep -vE "\.test\.|// |/\*|\.exec\(" || echo "NONE"
echo "--- http(s)/socket/dns ---"
grep -rnE "\b(fetch|axios|WebSocket|node:http|node:https|node:net|node:dns|createServer|\.listen\()" \
  src index.ts cli.ts | grep -vE "\.test\.|// " || echo "NONE"
echo "--- telemetry/analytics/update-check ---"
grep -rniE "sentry|posthog|analytics|update.?check" src index.ts cli.ts \
  | grep -vE "\.test\.|// |telemetry-store|compaction-telemetry" || echo "NONE"

echo ""
echo "==> [4/5] Air-gap hard block + full test suite"
cat > /tmp/lcm-airgap.cjs <<'EOF'
const dns = require('node:dns');
dns.lookup = () => { throw new Error('AIRGAP: DNS lookup'); };
dns.lookupService = () => { throw new Error('AIRGAP: lookupService'); };
for (const mod of ['https','http','net','tls']) {
  try {
    const m = require('node:'+mod);
    (m.request || m.connect || m.createConnection || m.connect) &&
      (m.request = m.get = m.connect = m.createConnection = () => { throw new Error('AIRGAP: '+mod); });
  } catch {}
}
EOF
NODE_OPTIONS="--require /tmp/lcm-airgap.cjs" npx vitest run 2>&1 | tail -6

echo ""
echo "==> [5/5] Shipped CLI offline smoke against real local DB"
NODE_OPTIONS="--require /tmp/lcm-airgap.cjs" node dist/cli.js status --pretty | head -12
NODE_OPTIONS="--require /tmp/lcm-airgap.cjs" node dist/cli.js summaries list --limit 1 --format table | head -3

echo ""
echo "==> DONE: evidence reproduced. See docs/security/*.md"
