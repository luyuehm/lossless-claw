import { existsSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { getFileBackedDatabasePath } from "../db/connection.js";
import { resolveOpenclawStateDir } from "../db/config.js";
import { buildLcmDatabaseBackupPath, writeLcmDatabaseBackup } from "./lcm-db-backup.js";

/**
 * Self-healing doctor for orphaned `context_engine_turn_outbox` rows.
 *
 * OpenClaw's core context-engine turn advancement persists accepted turns to a
 * per-agent outbox table (`context_engine_turn_outbox`) inside each agent's
 * `openclaw-agent.sqlite`. A row normally flows `admitted -> accepted -> ready`
 * and is deleted once the context engine commits the turn. When the gateway is
 * interrupted between those stages — or a session is tombstoned during restart
 * recovery — a row can be left permanently in `admitted` or `blocked` state.
 *
 * `admitted` rows are the harmful kind: OpenClaw's drain keeps reporting them
 * as pending (they are not `blocked`), so the host re-attempts them on every
 * turn, pinning the event loop at ~100% CPU. `blocked` rows are terminal but
 * are never purged by the core either. This doctor scans every agent database
 * and clears both kinds of orphan, which is exactly the fix the gateway
 * otherwise requires `openclaw doctor --fix` to perform manually.
 *
 * The scan is read-only and cross-database. Apply backs each affected database
 * up via `VACUUM INTO` before deleting, and refuses to run on memory databases.
 */

/** State values that mark a turn advancement as permanently orphaned. */
export const ORPHANED_ADVANCEMENT_STATES = ["admitted", "blocked"] as const;

export type OrphanedAdvancementState = (typeof ORPHANED_ADVANCEMENT_STATES)[number];

export type OrphanedAdvancementRow = {
  agentId: string;
  databasePath: string;
  advancementKey: string;
  state: OrphanedAdvancementState;
  failure: string | null;
  sessionId: string;
  attemptCount: number;
  createdAt: number | null;
};

export type OrphanedAdvancementScan = {
  databasesScanned: number;
  agentRows: Array<{
    agentId: string;
    databasePath: string;
    count: number;
    byState: Record<OrphanedAdvancementState, number>;
    examples: OrphanedAdvancementRow[];
  }>;
  totalOrphaned: number;
};

export type OrphanedAdvancementApplyResult =
  | {
      kind: "applied";
      agentId: string;
      databasePath: string;
      deletedRows: number;
      backupPath: string;
    }
  | {
      kind: "skipped";
      agentId: string;
      reason: string;
    };

/** Resolve the directory holding every agent's `openclaw-agent.sqlite`. */
function resolveAgentDatabasesDir(stateDir: string = resolveOpenclawStateDir()): string {
  return join(stateDir, "agents");
}

/** List agent ids that have a file-backed agent database on disk. */
export function listAgentDatabasePaths(
  stateDir: string = resolveOpenclawStateDir(),
): Array<{ agentId: string; databasePath: string }> {
  const agentsDir = resolveAgentDatabasesDir(stateDir);
  if (!existsSync(agentsDir)) {
    return [];
  }
  const results: Array<{ agentId: string; databasePath: string }> = [];
  for (const entry of readdirSync(agentsDir, { withFileTypes: true })) {
    if (!entry.isDirectory() && !entry.isSymbolicLink()) {
      continue;
    }
    const databasePath = join(agentsDir, entry.name, "agent", "openclaw-agent.sqlite");
    if (!existsSync(databasePath)) {
      continue;
    }
    results.push({ agentId: entry.name, databasePath });
  }
  return results.sort((a, b) => a.agentId.localeCompare(b.agentId));
}

function openAgentDatabaseReadOnly(databasePath: string): DatabaseSync {
  return new DatabaseSync(databasePath, { readOnly: true });
}

function openAgentDatabaseWritable(databasePath: string): DatabaseSync {
  const fileBacked = getFileBackedDatabasePath(databasePath);
  if (!fileBacked) {
    throw new Error(`refusing to open a non-file-backed agent database: ${databasePath}`);
  }
  return new DatabaseSync(fileBacked);
}

/** Read orphaned advancement rows for a single agent database. */
function scanAgentOrphanedAdvancements(databasePath: string): OrphanedAdvancementRow[] {
  let db: DatabaseSync | undefined;
  try {
    db = openAgentDatabaseReadOnly(databasePath);
  } catch {
    // A database the gateway has open in WAL mode may briefly refuse a second
    // reader; treat it as unscannable rather than failing the whole scan.
    return [];
  }
  try {
    const hasTable = db
      .prepare(
        `SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'context_engine_turn_outbox'`,
      )
      .get();
    if (!hasTable) {
      return [];
    }
    const rows = db
      .prepare(
        `SELECT advancement_key,
                json_extract(payload_json, '$.state') AS state,
                json_extract(payload_json, '$.failure') AS failure,
                session_id,
                attempt_count,
                created_at
           FROM context_engine_turn_outbox
          WHERE json_extract(payload_json, '$.state') IN ('admitted', 'blocked')
          ORDER BY created_at ASC`,
      )
      .all() as Array<{
      advancement_key: string;
      state: string;
      failure: string | null;
      session_id: string;
      attempt_count: number;
      created_at: number | null;
    }>;
    return rows
      .filter((row) => ORPHANED_ADVANCEMENT_STATES.includes(row.state as OrphanedAdvancementState))
      .map((row) => ({
        agentId: "",
        databasePath,
        advancementKey: row.advancement_key,
        state: row.state as OrphanedAdvancementState,
        failure: row.failure,
        sessionId: row.session_id,
        attemptCount: row.attempt_count,
        createdAt: row.created_at,
      }));
  } finally {
    db.close();
  }
}

/** Scan every agent database for orphaned turn advancements (read-only). */
export function scanOrphanedAdvancements(
  options: {
    stateDir?: string;
    agentIds?: string[];
  } = {},
): OrphanedAdvancementScan {
  const requested = new Set(
    (options.agentIds ?? []).map((agentId) => agentId.trim()).filter(Boolean),
  );
  const all = listAgentDatabasePaths(options.stateDir);
  const targets =
    requested.size > 0 ? all.filter((entry) => requested.has(entry.agentId)) : all;

  const agentRows: OrphanedAdvancementScan["agentRows"] = [];
  let totalOrphaned = 0;
  for (const { agentId, databasePath } of targets) {
    const rows = scanAgentOrphanedAdvancements(databasePath).map((row) => ({
      ...row,
      agentId,
    }));
    if (rows.length === 0) {
      continue;
    }
    const byState: Record<OrphanedAdvancementState, number> = {
      admitted: 0,
      blocked: 0,
    };
    for (const row of rows) {
      byState[row.state] += 1;
    }
    totalOrphaned += rows.length;
    agentRows.push({
      agentId,
      databasePath,
      count: rows.length,
      byState,
      examples: rows.slice(0, 5),
    });
  }

  return {
    databasesScanned: targets.length,
    agentRows,
    totalOrphaned,
  };
}

/** Delete orphaned advancements for a single agent database, after a backup. */
function applyAgentOrphanedAdvancements(
  agentId: string,
  databasePath: string,
): OrphanedAdvancementApplyResult {
  const fileBacked = getFileBackedDatabasePath(databasePath);
  if (!fileBacked) {
    return {
      kind: "skipped",
      agentId,
      reason: "agent database is not file-backed; refusing to modify",
    };
  }

  const backupPath = buildLcmDatabaseBackupPath(databasePath, "doctor-orphaned-advancements");
  if (!backupPath) {
    return { kind: "skipped", agentId, reason: "could not resolve a backup path" };
  }

  let db: DatabaseSync | undefined;
  try {
    db = openAgentDatabaseWritable(databasePath);
    const hasTable = db
      .prepare(
        `SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'context_engine_turn_outbox'`,
      )
      .get();
    if (!hasTable) {
      return { kind: "skipped", agentId, reason: "no context_engine_turn_outbox table" };
    }

    // Only write a backup when there is actually something to delete; the scan
    // already established that, but re-check under the writable handle.
    const pending = db
      .prepare(
        `SELECT COUNT(*) AS count
           FROM context_engine_turn_outbox
          WHERE json_extract(payload_json, '$.state') IN ('admitted', 'blocked')`,
      )
      .get() as { count: number };
    if (pending.count === 0) {
      return { kind: "skipped", agentId, reason: "no orphaned advancements to delete" };
    }

    writeLcmDatabaseBackup(db, backupPath);

    const result = db
      .prepare(
        `DELETE FROM context_engine_turn_outbox
          WHERE json_extract(payload_json, '$.state') IN ('admitted', 'blocked')`,
      )
      .run();
    const deletedRows = Number(result.changes ?? 0);

    return {
      kind: "applied",
      agentId,
      databasePath,
      deletedRows,
      backupPath,
    };
  } finally {
    db?.close();
  }
}

/** Apply orphaned advancement cleanup across agent databases. */
export function applyOrphanedAdvancements(
  options: {
    stateDir?: string;
    agentIds?: string[];
  } = {},
): OrphanedAdvancementApplyResult[] {
  const requested = new Set(
    (options.agentIds ?? []).map((agentId) => agentId.trim()).filter(Boolean),
  );
  const all = listAgentDatabasePaths(options.stateDir);
  const targets =
    requested.size > 0 ? all.filter((entry) => requested.has(entry.agentId)) : all;

  const results: OrphanedAdvancementApplyResult[] = [];
  for (const { agentId, databasePath } of targets) {
    results.push(applyAgentOrphanedAdvancements(agentId, databasePath));
  }
  return results;
}
