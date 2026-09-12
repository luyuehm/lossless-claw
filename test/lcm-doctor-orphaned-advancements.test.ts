import { mkdtempSync, mkdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { describe, expect, it, afterEach } from "vitest";
import {
  applyOrphanedAdvancements,
  scanOrphanedAdvancements,
  type OrphanedAdvancementState,
} from "../src/plugin/lcm-doctor-orphaned-advancements.js";

const OUTBOX_SCHEMA = `
CREATE TABLE context_engine_turn_outbox (
  advancement_key TEXT NOT NULL PRIMARY KEY,
  engine_id TEXT NOT NULL,
  owner_plugin_id TEXT,
  session_id TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  attempt_count INTEGER NOT NULL DEFAULT 0,
  last_attempt_at INTEGER,
  last_error TEXT,
  created_at INTEGER NOT NULL
) STRICT;
`;

const cleanupDirs: string[] = [];

function makeAgentDb(agentId: string, rows: Array<{ state: OrphanedAdvancementState; failure?: string }>): string {
  const root = mkdtempSync(join(tmpdir(), "lcm-orphaned-"));
  cleanupDirs.push(root);
  const agentDir = join(root, "agents", agentId, "agent");
  mkdirSync(agentDir, { recursive: true });
  const dbPath = join(agentDir, "openclaw-agent.sqlite");
  const db = new DatabaseSync(dbPath);
  db.exec(OUTBOX_SCHEMA);
  const insert = db.prepare(
    `INSERT INTO context_engine_turn_outbox (advancement_key, engine_id, session_id, payload_json, created_at)
     VALUES (?, 'lossless-claw', ?, ?, ?)`,
  );
  rows.forEach((row, index) => {
    const payload: Record<string, unknown> = {
      state: row.state,
      admission: {
        agentId,
        sessionId: `session-${index}`,
        sessionKey: `agent:${agentId}:telegram:direct:${index}`,
      },
    };
    if (row.failure !== undefined) {
      payload.failure = row.failure;
    }
    insert.run(`key-${index}`, `session-${index}`, JSON.stringify(payload), Date.now() + index);
  });
  db.close();
  return root;
}

afterEach(() => {
  for (const dir of cleanupDirs.splice(0)) {
    rmSync(dir, { recursive: true, force: true });
  }
});

describe("scanOrphanedAdvancements", () => {
  it("returns empty scan when no agent databases exist", () => {
    const root = mkdtempSync(join(tmpdir(), "lcm-orphaned-empty-"));
    cleanupDirs.push(root);
    const scan = scanOrphanedAdvancements({ stateDir: root });
    expect(scan.totalOrphaned).toBe(0);
    expect(scan.agentRows).toEqual([]);
  });

  it("detects admitted and blocked orphaned rows across agents", () => {
    const rootA = makeAgentDb("agent-a", [
      { state: "admitted" },
      { state: "blocked", failure: "stale" },
    ]);
    const scanA = scanOrphanedAdvancements({ stateDir: rootA });
    expect(scanA.totalOrphaned).toBe(2);
    expect(scanA.agentRows).toHaveLength(1);
    expect(scanA.agentRows[0].byState.admitted).toBe(1);
    expect(scanA.agentRows[0].byState.blocked).toBe(1);
  });

  it("scans across multiple agent databases", () => {
    const root = mkdtempSync(join(tmpdir(), "lcm-orphaned-multi-"));
    cleanupDirs.push(root);
    const agentsDir = join(root, "agents");
    mkdirSync(agentsDir, { recursive: true });
    const fixtures: Array<[string, Array<{ state: OrphanedAdvancementState; failure?: string }>]> = [
      ["agent-x", [{ state: "admitted" }]],
      ["agent-y", [{ state: "blocked", failure: "stale" }]],
    ];
    for (const [agentId, rows] of fixtures) {
      const agentDir = join(agentsDir, agentId, "agent");
      mkdirSync(agentDir, { recursive: true });
      const db = new DatabaseSync(join(agentDir, "openclaw-agent.sqlite"));
      db.exec(OUTBOX_SCHEMA);
      const insert = db.prepare(
        `INSERT INTO context_engine_turn_outbox (advancement_key, engine_id, session_id, payload_json, created_at)
         VALUES (?, 'lossless-claw', ?, ?, ?)`,
      );
      rows.forEach((row, index) => {
        const payload: Record<string, unknown> = {
          state: row.state,
          admission: { agentId, sessionId: `s-${index}`, sessionKey: `agent:${agentId}:x:${index}` },
        };
        if (row.failure !== undefined) payload.failure = row.failure;
        insert.run(`key-${index}`, `s-${index}`, JSON.stringify(payload), Date.now() + index);
      });
      db.close();
    }
    const scan = scanOrphanedAdvancements({ stateDir: root });
    expect(scan.totalOrphaned).toBe(2);
    expect(scan.agentRows).toHaveLength(2);
  });

  it("ignores ready/accepted rows", () => {
    const root = makeAgentDb("agent-b", [{ state: "admitted" }]);
    // Append a ready row that should NOT be counted.
    const dbPath = join(root, "agents", "agent-b", "agent", "openclaw-agent.sqlite");
    const db = new DatabaseSync(dbPath);
    db.prepare(
      `INSERT INTO context_engine_turn_outbox (advancement_key, engine_id, session_id, payload_json, created_at)
       VALUES ('ready-key', 'lossless-claw', 'ready-session', '{"state":"ready"}', 0)`,
    ).run();
    db.close();
    const scan = scanOrphanedAdvancements({ stateDir: root });
    expect(scan.totalOrphaned).toBe(1);
    expect(scan.agentRows[0].byState.admitted).toBe(1);
    expect(scan.agentRows[0].byState.blocked).toBe(0);
  });
});

describe("applyOrphanedAdvancements", () => {
  it("deletes orphaned rows and backs up the database", () => {
    const root = makeAgentDb("agent-c", [
      { state: "admitted" },
      { state: "blocked", failure: "session-rebound" },
    ]);
    const results = applyOrphanedAdvancements({ stateDir: root });
    expect(results).toHaveLength(1);
    const result = results[0];
    expect(result.kind).toBe("applied");
    if (result.kind === "applied") {
      expect(result.deletedRows).toBe(2);
      expect(result.backupPath).toContain("doctor-orphaned-advancements");
    }

    const after = scanOrphanedAdvancements({ stateDir: root });
    expect(after.totalOrphaned).toBe(0);
  });

  it("returns empty results when nothing to clean", () => {
    const root = mkdtempSync(join(tmpdir(), "lcm-orphaned-none-"));
    cleanupDirs.push(root);
    const agentsDir = join(root, "agents");
    mkdirSync(agentsDir, { recursive: true });
    const results = applyOrphanedAdvancements({ stateDir: root });
    expect(results).toEqual([]);
  });
});
