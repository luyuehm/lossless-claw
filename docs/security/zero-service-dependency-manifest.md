# lossless-claw 零服务依赖清单（Zero-Service Dependency Manifest）

> 基线：`v1.0.0`（`c52acc0 merge: official v1.0.0 with fork fixes assessed`）
> 审计日期：2026-09-15
> 审计对象：`src/`（105 个 .ts 文件）+ 产物 `dist/*.js` + 配置面 `openclaw.plugin.json`
> 目标：把「本地优先 + 零服务依赖」从声明变成可审计事实。

## 结论

lossless-claw v1.0.0 **无任何主动外部调用点**。全插件对外依赖为零：

- 运行时依赖仅 **1 个纯本地包**（`@sinclair/typebox`，schema 校验，无 I/O）；
- 其余全部是 Node 内置模块（`node:*`）与 **OpenClaw 宿主机进程内插件 SDK**（`openclaw/plugin-sdk/*`，进程内调用，非网络服务）；
- 无 HTTP 客户端、无 socket、无 DNS 主动查询、无 child_process 外呼、无 telemetry/analytics/update-check；
- 模型调用 100% 走 `request_api: "runtime.llm"` — 由宿主 OpenClaw 进程内提供，插件自身不做任何 API 直连。

---

## 1. 依赖面（本地可跑的完整面）

### 1.1 npm 依赖

| 包 | 类型 | 用途 | 网络? |
|---|---|---|---|
| `@sinclair/typebox` 0.34.48 | dependencies（唯一） | JSON schema 校验 | 无 |
| `openclaw` | peerDependencies (optional) | 宿主进程内插件 SDK | 无（进程内） |

devDependencies（`esbuild`/`typescript`/`vitest`/`@earendil-works/*`/`@changesets/*`）仅构建与测试用，**不随 `npm pack` 发布**（见 §4）。

### 1.2 模块导入面（`src/` 全部 import）

对 `src/`、`index.ts`、`cli.ts` 全部 import 去重统计：

- `node:*` 内置：`node:sqlite`(34)、`node:crypto`(19)、`node:path`(18)、`node:fs`(17)、`node:os`(3)、`node:module`(3)、`node:util`(1)、`node:readline`(1)、`node:async_hooks`(1) — **全为本地能力**；
- 裸包：仅 `@sinclair/typebox`、`os`、`path`；
- 宿主进程内 SDK：`openclaw/plugin-sdk/logging-core`、`openclaw/plugin-sdk/session-transcript-runtime`、`openclaw/plugin-sdk/core` — 通过 `require()`/动态 `import()` 延迟加载，进程内；
- **无** `node:http` / `node:https` / `node:net` / `node:dns` / `node:http2` / `node:tls` / `node:child_process` / `node:worker_threads`。

### 1.3 产物 bundle 面（发布内容）

对 `npm run build` 产物逐一核对：

| Bundle | 外部导入 | 网络/服务导入 |
|---|---|---|
| `dist/index.js` (1.1MB) | 仅 `node:*` + `import("openclaw/plugin-sdk/session-transcript-runtime")` | **0** |
| `dist/cli.js` (77KB) | 仅 `node:*` | **0** |
| `dist/migrate-sessions.js` (182KB) | 仅 `node:*` | **0** |

### 1.4 配置面（`openclaw.plugin.json` + `src/db/config.ts`）

全部配置项均为**本地行为调优**（阈值、token 预算、路径、模型选择）。含网络语义的字段：
`summaryProvider` / `summaryModel` / `expansionProvider` / `expansionModel` 等，全部仅为「把模型选择交给宿主 runtime.llm 的提示」，插件不发起任何 API 请求。

环境变量引用（`process.env.*`）全集：
`OPENCLAW_STATE_DIR`（本地状态路径）、`LCM_SUMMARY_MODEL` / `LCM_SUMMARY_PROVIDER` / `OPENCLAW_PROVIDER`（本地模型选择提示）、`VITEST`（测试标记）。**无任何 URL / endpoint / token 环境变量。**

---

## 2. 可选能力与默认开关

审计确认：**不存在任何「可选联网能力」需要关闭**。仓库内出现的两处网络 I/O 均为**非发布件**：

| 位置 | 内容 | 是否发布 | 处置 |
|---|---|---|---|
| `tui/`（Go） | `net/http` 作为 LLM **客户端**（调 `api.anthropic.com`/`api.openai.com` 或自定义 `--base-url`） | ❌ 不在 `package.json "files"`；不随 npm 发布 | 豁免：管理端工具，非插件生命周期，需手动构建运行 |
| `scripts/stub-tier-drilldown-harness.mjs` | 开发用 harness，`fetch("https://openrouter.ai/api/v1/chat/completions")` | ❌ 不在 `"files"`；dev 脚本 | 豁免：开发/回归工具，非发布件 |

插件运行时（`dist/`）**不含** TUI 与 scripts。断网实测（§5）在**不触碰、不调用**上述两项的前提下通过，证明插件生命周期本身零联网。

**强依赖阈值**：插件运行唯一强依赖是 **OpenClaw 宿主**（peerDependencies，进程内 SDK 与 `runtime.llm`）。该宿主为本地进程（`openclaw gateway` 本地运行），**不是外部服务**。除此之外无中间件、无数据库服务（用 `node:sqlite` 本地文件库）、无队列。

---

## 3. 外部调用点审计结论（逐项）

扫描方法：全量 grep `src/` + 产物 bundle（fetch/axios/WebSocket/net/http(s)/dns/createServer/socket/child_process/spawn/exec/telemetry/analytics/update-check/URL 字面量）。

| # | 检查项 | 结果 |
|---|---|---|
| 1 | HTTP/HTTPS 客户端 | 0（bundle 与 src 均无） |
| 2 | WebSocket / socket / dgram | 0 |
| 3 | DNS 主动查询 | 0 |
| 4 | child_process / spawn / exec 外呼 | 0（`db.exec` 为 SQLite 语句执行） |
| 5 | Telemetry / analytics / sentry / posthog | 0（`compaction-telemetry` 为本地 SQLite 落盘，非外传） |
| 6 | Update-check / 版本自检联网 | 0 |
| 7 | URL 字面量 | 仅 3 处注释/文档链接（github issue/PR）与无网络语义的常量 |
| 8 | 环境变量含网络端点 | 0 |
| 9 | 大文件/附件外传 | 0（`largeFilesDir` 本地磁盘） |

**未发现需要「关闭」的调用点，也无豁免的运行时调用点。** 所有存储（SQLite DAG、FTS5、large-file 侧车、独立日志）均为本地文件。

---

## 4. 发布件与仓库件边界（证据文件清单）

`npm pack --dry-run`（v1.0.0）确认 tarball 仅含 20 个文件：
`LICENSE`、`README.md`、`docs/*.md`（8）、`doctor-contract-api.{d.ts,js}`、`openclaw.plugin.json`、`package.json`、`skills/lossless-claw/**`（7）、`dist/`（构建产物）。

**排除项**：`src/`、`test/`、`tui/`、`scripts/`、`audit/`、`architecture/`、`node_modules/`。→ TUI 与 harness 的网络调用**不会进入发布包**。

### 证据文件清单

| 证据 | 路径 |
|---|---|
| 构建产物（含 0 网络导入） | `dist/index.js`、`dist/cli.js`、`dist/migrate-sessions.js` |
| 配置 schema（全本地调优） | `openclaw.plugin.json` |
| 宿主进程内 SDK 边界 | `src/openclaw-bridge.ts`、`src/plugin/index.ts` |
| LLM 委托（`request_api: "runtime.llm"`） | `src/plugin/index.ts:1325` |
| 本地 SQLite/FTS/大文件存储 | `src/db/`、`src/store/`、`src/large-files.ts` |
| 确定性回退摘要（无 LLM 可跑） | `src/summary-fallback.ts` |
| 离线测试证据 | 见 `docs/security/air-gap-declaration.md` §5 |
| 可复现证据脚本 | `scripts/airgap-evidence.sh` |
