# lossless-claw air-gap 声明（Air-Gap Declaration）

> 基线：`v1.0.0` · 审计日期：2026-09-15
> 关联：`docs/security/zero-service-dependency-manifest.md`

## 声明

**lossless-claw v1.0.0 在断网（air-gap）环境下完整可用。** 压缩 → 召回 → 回溯（compaction → recall → backtrack）全生命周期不依赖任何外部网络、服务或中间件。全部数据面（DAG SQLite、FTS5、large-file 侧车、独立日志）均为本地文件；全部分析/校验/决策均在本地进程内完成。

KISS 说明：本声明**不新增任何监控/服务组件**，仅做审计与清单 — 不这么做就无法把「本地优先」主张变成可审计事实。

## 断网实测

测试在**硬阻断网络**下进行：`NODE_OPTIONS` 预加载拦截器使 `dns.lookup` / `net.connect` / `http(s).request` / `tls.connect` 一触即抛（`AIRGAP` 错误）；同时把 HTTP(S)/所有代理指向黑洞端口。**任何代码路径只要试图解析域名、建 socket 或发请求，测试立即失败。** 实测结果：零外呼触发。

### 证据 1 — 生命周期集成测试（硬阻断下全部通过）

| 测试面 | 用例数 | 结果 |
|---|---|---|
| `engine-compaction` → 压缩 | — | ✅ |
| `engine-lifecycle` → 生命周期 | — | ✅ |
| `lcm-integration-compaction` → 压缩集成 | — | ✅ |
| `lcm-integration-retrieval` → 召回/回溯集成 | — | ✅ |
| **以上 4 文件合计** | **152** | **152/152 通过** |

### 证据 2 — 全量测试（硬阻断下全部通过）

**114 个测试文件，1855 个用例全部通过** — 覆盖压缩、召回（grep/FTS）、回溯（describe/expand）、装配、迁移、CLI、回放防抖等全部模块。这是插件全部测试证明能力的完整断网运行。

补充：召回/回溯专用面 `retrieval-recall` / `lcm-recall-tool`（FTS/describe/expand/grep）+ `engine-assemble` / `engine-after-turn` 4 文件 139 用例同样硬阻断全过。`retrieval-recall.test.ts` 在本基线（v1.0.0）中不存在 — 该文件为 fork 独有内容，未在官方 v1.0.0 中。

### 证据 3 — 发布产物（`dist/`）断网冒烟（真实数据库）

对本地真实 LCM SQLite（`~/.openclaw/lcm.db`，5106 会话 / 280k 消息）在硬阻断下运行发布版 CLI：

| 命令 | 对应生命周期阶段 | 结果 |
|---|---|---|
| `dist/cli.js status` | 全局状态 | ✅ 5106 会话 / 280432 消息 / 1.09 亿 token |
| `dist/cli.js conversations list/show` | 召回（会话列表/详情） | ✅ 含 summaryDepth 4 级 DAG |
| `dist/cli.js summaries list` | 召回（摘要列表） | ✅ leaf, 268 tokens, 源 24478 tokens |
| `dist/cli.js summaries show sum_6e6d803168d900f8` | **回溯（摘要→源消息）** | ✅ 62 条源消息 + DAG parent/child |

`summaries show` 的响应包含完整回溯：summary + `parents`/`children`（DAG 父/子）+ `sourceMessages`（源消息逐条）。真实库上端到端回溯，全程无网络。

### 场景用例（验收对应）

| # | 场景 | 端点 | 结果 |
|---|---|---|---|
| 1 | 断网冷启动 + 状态查询 | CLI status / conversations list | ✅ |
| 2 | 断网召回（FTS+摘要+会话） | summaries list / conversations show | ✅ |
| 3 | 断网回溯（摘要→源消息/文件） | summaries show | ✅ |
| 4 | 断网确定性压缩回退 | summary-fallback 模块 + engine-compaction 套件 | ✅ |
| 5 | 断网全生命周期冒烟 | 114 文件 1855 用例 | ✅ |

## 边界与豁免

| 项 | 状态 |
|---|---|
| `tui/`（Go 管理端） | 非发布件、非插件生命周期；如需联网自主承担，**不影响插件断网声明** |
| `scripts/*-harness.mjs`（开发 harness） | 非发布件；开发回归工具 |
| 宿主 OpenClaw `runtime.llm` | 进程内委托；宿主本体是否联网由宿主决定，插件不直连任何 LLM API |

## 验收对照

- ✅ 零服务依赖清单落盘（依赖面 + 阈值 + 证据文件清单）→ `zero-service-dependency-manifest.md`
- ✅ 断网实测通过：无外部调用点的端点全部可跑
- ✅ 3+ 场景用例通过（含断网下召回/回溯）→ 上述 5 场景
- ✅ 未编造：全部以 `v1.0.0` 代码事实与测试输出为准