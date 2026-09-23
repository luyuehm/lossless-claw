package main

import (
	"database/sql"
	"path/filepath"
	"testing"

	_ "modernc.org/sqlite"
)

func TestMemoryHealthAndEvidenceChains(t *testing.T) {
	path := filepath.Join(t.TempDir(), "lcm.db")
	db, err := sql.Open("sqlite", path)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	schema := `
CREATE TABLE summaries(summary_id TEXT PRIMARY KEY, conversation_id INTEGER, kind TEXT, token_count INTEGER);
CREATE TABLE messages(message_id INTEGER PRIMARY KEY, role TEXT, content TEXT, created_at TEXT, token_count INTEGER);
CREATE TABLE summary_messages(summary_id TEXT, message_id INTEGER, ordinal INTEGER);
CREATE TABLE pending_summary_nodes(node_id TEXT, conversation_id INTEGER, status TEXT);`
	if _, err := db.Exec(schema); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Exec(`INSERT INTO summaries VALUES ('leaf-1',7,'leaf',20),('leaf-2',7,'leaf',20),('root',7,'condensed',10); INSERT INTO messages VALUES (101,'user','alpha','2026-01-01',100),(102,'assistant','beta','2026-01-02',100),(103,'user','gamma','2026-01-03',100); INSERT INTO summary_messages VALUES ('leaf-1',101,0),('leaf-1',102,1),('leaf-1',103,2); INSERT INTO pending_summary_nodes VALUES ('p1',7,'planned'),('p2',7,'promoted');`); err != nil {
		t.Fatal(err)
	}

	h, err := loadMemoryHealth(path, 7)
	if err != nil {
		t.Fatal(err)
	}
	if h.leafCount != 2 || h.condensedCount != 1 || h.pendingCount != 1 {
		t.Fatalf("unexpected health: %+v", h)
	}
	if h.compressionPct != 16 {
		t.Fatalf("expected 16%% compression ratio, got %d", h.compressionPct)
	}

	sources, err := loadSummarySources(path, "leaf-1")
	if err != nil {
		t.Fatal(err)
	}
	if len(sources) != 3 {
		t.Fatalf("expected 3 evidence links, got %d", len(sources))
	}
	for i, want := range []int64{101, 102, 103} {
		if sources[i].id != want {
			t.Errorf("evidence %d: want message %d, got %d", i, want, sources[i].id)
		}
	}
}

func TestMemoryHealthSupportsDatabaseWithoutPendingTable(t *testing.T) {
	path := filepath.Join(t.TempDir(), "lcm.db")
	db, err := sql.Open("sqlite", path)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	_, err = db.Exec(`CREATE TABLE summaries(summary_id TEXT PRIMARY KEY, conversation_id INTEGER, kind TEXT, token_count INTEGER); CREATE TABLE messages(message_id INTEGER PRIMARY KEY, token_count INTEGER); CREATE TABLE summary_messages(summary_id TEXT, message_id INTEGER); INSERT INTO summaries VALUES ('s',1,'leaf',5);`)
	if err != nil {
		t.Fatal(err)
	}
	h, err := loadMemoryHealth(path, 1)
	if err != nil {
		t.Fatal(err)
	}
	if h.pendingCount != 0 {
		t.Fatalf("expected zero pending nodes, got %d", h.pendingCount)
	}
}
