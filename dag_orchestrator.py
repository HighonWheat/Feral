"""
Phase 6, Step 1 - SQLite Hierarchical Task DAG Orchestrator (`dag_orchestrator.py`)
===================================================================================
Provides Feral with a persistent, crash-recoverable Directed Acyclic Graph (DAG) task
scheduler backed by SQLite (`WAL` mode) for long-running autonomous missions and the
Phase 7 Odysseus-inspired Autonomous Agents Board & Mission Control UI:

  1. Durable SQLite Graph Storage (`missions`, `dag_nodes`, `dag_edges`):
     Every mission and task node (`PENDING`, `READY`, `RUNNING`, `COMPLETED`, `FAILED`,
     `BLOCKED`) is persisted atomically on disk so multi-hour workflows survive restarts.
  2. Topological Readiness & Cycle Guard (`add_dependency`, `refresh_ready_states`):
     Validates that the dependency graph remains strictly acyclic (DFS cycle check) and
     automatically promotes `PENDING` nodes to `READY` when all upstream parent nodes
     reach `COMPLETED`.
  3. Strike-Counted Retry & Downstream `BLOCKED` Propagation:
     When a node fails an execution attempt (`mark_node_failed`), it automatically re-queues
     as `READY` while `attempts < max_retries` (e.g., 3 strikes). If it exhausts `max_retries`,
     it transitions to `FAILED` and cascades `BLOCKED` to all dependent downstream nodes.
  4. Crash-Safe Restart Recovery (`recover_interrupted_nodes`):
     On startup, detects any nodes left in `RUNNING` state by an interrupted session and
     resets them to `READY` while preserving all `COMPLETED` upstream results.
"""
import contextlib
import json
import shutil
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple

from feral_core import ROOT

DEFAULT_DB_PATH = ROOT / "outputs" / "feral_dag.sqlite"


@dataclass
class DAGNode:
    node_id: str
    mission_id: str
    title: str
    prompt: str
    assigned_agent: str
    status: str  # PENDING | READY | RUNNING | COMPLETED | FAILED | BLOCKED
    priority: int
    attempts: int
    max_retries: int
    result_summary: str
    error_message: str
    dependencies: List[str] = field(default_factory=list)
    updated_at: float = 0.0


class DAGOrchestrator:
    """
    Persistent SQLite Task DAG Orchestrator for Feral's multi-step autonomous workflows.
    """

    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextlib.contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path), timeout=15.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS missions (
                    mission_id TEXT PRIMARY KEY,
                    goal TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'ACTIVE',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS dag_nodes (
                    node_id TEXT NOT NULL,
                    mission_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    assigned_agent TEXT NOT NULL DEFAULT 'Feral-Core',
                    status TEXT NOT NULL DEFAULT 'PENDING',
                    priority INTEGER NOT NULL DEFAULT 10,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    max_retries INTEGER NOT NULL DEFAULT 3,
                    result_summary TEXT NOT NULL DEFAULT '',
                    error_message TEXT NOT NULL DEFAULT '',
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (mission_id, node_id),
                    FOREIGN KEY (mission_id) REFERENCES missions(mission_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS dag_edges (
                    mission_id TEXT NOT NULL,
                    parent_node_id TEXT NOT NULL,
                    child_node_id TEXT NOT NULL,
                    PRIMARY KEY (mission_id, parent_node_id, child_node_id)
                );
                """
            )

    def create_mission(self, mission_id: str, goal: str) -> None:
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO missions (mission_id, goal, status, created_at, updated_at) VALUES (?, ?, 'ACTIVE', ?, ?)",
                (mission_id, goal, now, now),
            )

    def add_node(
        self,
        mission_id: str,
        node_id: str,
        title: str,
        prompt: str,
        assigned_agent: str = "Feral-Core",
        depends_on: Optional[List[str]] = None,
        priority: int = 10,
        max_retries: int = 3,
    ) -> DAGNode:
        now = time.time()
        deps = list(depends_on or [])
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO dag_nodes
                (node_id, mission_id, title, prompt, assigned_agent, status, priority, attempts, max_retries, result_summary, error_message, updated_at)
                VALUES (?, ?, ?, ?, ?, 'PENDING', ?, 0, ?, '', '', ?)
                """,
                (node_id, mission_id, title, prompt, assigned_agent, int(priority), int(max_retries), now),
            )
        for parent_id in deps:
            self.add_dependency(mission_id, parent_node_id=parent_id, child_node_id=node_id)
        self.refresh_ready_states(mission_id)
        return self.get_node(mission_id, node_id)

    def _has_path(self, conn: sqlite3.Connection, mission_id: str, start: str, target: str) -> bool:
        """DFS reachability check: returns True if `target` is reachable from `start`."""
        visited: Set[str] = set()
        stack = [start]
        while stack:
            cur = stack.pop()
            if cur == target:
                return True
            if cur in visited:
                continue
            visited.add(cur)
            rows = conn.execute(
                "SELECT child_node_id FROM dag_edges WHERE mission_id = ? AND parent_node_id = ?",
                (mission_id, cur),
            ).fetchall()
            for r in rows:
                stack.append(r["child_node_id"])
        return False

    def add_dependency(self, mission_id: str, parent_node_id: str, child_node_id: str) -> None:
        """Adds edge `parent_node_id -> child_node_id`, raising `ValueError` if it creates a cycle."""
        if parent_node_id == child_node_id:
            raise ValueError(f"Self-dependency cycle detected on node `{parent_node_id}`")

        with self._connect() as conn:
            if self._has_path(conn, mission_id, start=child_node_id, target=parent_node_id):
                raise ValueError(
                    f"Cycle detected in DAG `{mission_id}`: adding `{parent_node_id} -> {child_node_id}` creates a loop"
                )
            conn.execute(
                "INSERT OR IGNORE INTO dag_edges (mission_id, parent_node_id, child_node_id) VALUES (?, ?, ?)",
                (mission_id, parent_node_id, child_node_id),
            )
        self.refresh_ready_states(mission_id)

    def refresh_ready_states(self, mission_id: str) -> None:
        """
        Re-evaluates every `PENDING` or `BLOCKED` node in `mission_id`:
          - If any upstream parent is `FAILED` or `BLOCKED` -> child becomes `BLOCKED`.
          - Else if all upstream parents are `COMPLETED` (or node has 0 parents) -> child becomes `READY`.
          - Else -> child remains `PENDING`.
        """
        now = time.time()
        with self._connect() as conn:
            nodes = conn.execute(
                "SELECT node_id, status FROM dag_nodes WHERE mission_id = ?",
                (mission_id,),
            ).fetchall()
            status_map: Dict[str, str] = {r["node_id"]: r["status"] for r in nodes}

            edges = conn.execute(
                "SELECT parent_node_id, child_node_id FROM dag_edges WHERE mission_id = ?",
                (mission_id,),
            ).fetchall()
            parents_of: Dict[str, List[str]] = {nid: [] for nid in status_map}
            for e in edges:
                parents_of.setdefault(e["child_node_id"], []).append(e["parent_node_id"])

            # Propagate readiness & blocking iteratively until fixed point
            changed = True
            while changed:
                changed = False
                for nid, cur_st in list(status_map.items()):
                    if cur_st in ("RUNNING", "COMPLETED", "FAILED"):
                        continue
                    p_ids = parents_of.get(nid, [])
                    p_statuses = [status_map[p] for p in p_ids if p in status_map]

                    if any(ps in ("FAILED", "BLOCKED") for ps in p_statuses):
                        new_st = "BLOCKED"
                    elif all(ps == "COMPLETED" for ps in p_statuses):
                        new_st = "READY"
                    else:
                        new_st = "PENDING"

                    if new_st != cur_st:
                        status_map[nid] = new_st
                        conn.execute(
                            "UPDATE dag_nodes SET status = ?, updated_at = ? WHERE mission_id = ? AND node_id = ?",
                            (new_st, now, mission_id, nid),
                        )
                        changed = True

    def recover_interrupted_nodes(self, mission_id: str) -> int:
        """
        Crash-recovery hook: resets any orphaned `RUNNING` nodes back to `READY` on startup
        while keeping all `COMPLETED` nodes intact.
        """
        now = time.time()
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE dag_nodes SET status = 'READY', updated_at = ? WHERE mission_id = ? AND status = 'RUNNING'",
                (now, mission_id),
            )
            recovered = cur.rowcount
        self.refresh_ready_states(mission_id)
        return recovered

    def get_node(self, mission_id: str, node_id: str) -> DAGNode:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM dag_nodes WHERE mission_id = ? AND node_id = ?",
                (mission_id, node_id),
            ).fetchone()
            if row is None:
                raise KeyError(f"Node `{node_id}` not found in mission `{mission_id}`")
            deps = [
                r["parent_node_id"]
                for r in conn.execute(
                    "SELECT parent_node_id FROM dag_edges WHERE mission_id = ? AND child_node_id = ? ORDER BY parent_node_id",
                    (mission_id, node_id),
                ).fetchall()
            ]
            return DAGNode(
                node_id=row["node_id"],
                mission_id=row["mission_id"],
                title=row["title"],
                prompt=row["prompt"],
                assigned_agent=row["assigned_agent"],
                status=row["status"],
                priority=row["priority"],
                attempts=row["attempts"],
                max_retries=row["max_retries"],
                result_summary=row["result_summary"],
                error_message=row["error_message"],
                dependencies=deps,
                updated_at=row["updated_at"],
            )

    def acquire_next_ready_node(self, mission_id: str) -> Optional[DAGNode]:
        """Atomically claims the highest-priority `READY` node, increments `attempts`, and sets `RUNNING`."""
        self.refresh_ready_states(mission_id)
        now = time.time()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT node_id FROM dag_nodes
                WHERE mission_id = ? AND status = 'READY'
                ORDER BY priority ASC, node_id ASC
                LIMIT 1
                """,
                (mission_id,),
            ).fetchone()
            if row is None:
                return None
            nid = row["node_id"]
            conn.execute(
                """
                UPDATE dag_nodes
                SET status = 'RUNNING', attempts = attempts + 1, updated_at = ?
                WHERE mission_id = ? AND node_id = ?
                """,
                (now, mission_id, nid),
            )
        return self.get_node(mission_id, nid)

    def mark_node_completed(self, mission_id: str, node_id: str, result_summary: str) -> DAGNode:
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE dag_nodes
                SET status = 'COMPLETED', result_summary = ?, error_message = '', updated_at = ?
                WHERE mission_id = ? AND node_id = ?
                """,
                (result_summary, now, mission_id, node_id),
            )
        self.refresh_ready_states(mission_id)
        return self.get_node(mission_id, node_id)

    def mark_node_failed(self, mission_id: str, node_id: str, error_message: str) -> DAGNode:
        """
        Records a failure on `node_id`. If `attempts < max_retries`, re-queues as `READY` for
        automatic self-healing retry; otherwise transitions to `FAILED` and blocks dependents.
        """
        node = self.get_node(mission_id, node_id)
        now = time.time()
        next_status = "READY" if node.attempts < node.max_retries else "FAILED"
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE dag_nodes
                SET status = ?, error_message = ?, updated_at = ?
                WHERE mission_id = ? AND node_id = ?
                """,
                (next_status, error_message, now, mission_id, node_id),
            )
        self.refresh_ready_states(mission_id)
        return self.get_node(mission_id, node_id)

    def get_upstream_context(self, mission_id: str, node_id: str) -> str:
        """Collects `result_summary` from all direct upstream dependencies to feed into a child node."""
        node = self.get_node(mission_id, node_id)
        if not node.dependencies:
            return ""
        lines = ["Upstream Dependency Outputs:"]
        for pid in node.dependencies:
            pnode = self.get_node(mission_id, pid)
            lines.append(f"- [{pid} | {pnode.title}]: {pnode.result_summary}")
        return "\n".join(lines)

    def get_dag_snapshot(self, mission_id: str) -> Dict[str, Any]:
        """Returns a complete JSON snapshot + Mermaid graph for the Phase 7 UI."""
        self.refresh_ready_states(mission_id)
        with self._connect() as conn:
            m_row = conn.execute("SELECT * FROM missions WHERE mission_id = ?", (mission_id,)).fetchone()
            n_rows = conn.execute(
                "SELECT node_id FROM dag_nodes WHERE mission_id = ? ORDER BY priority ASC, node_id ASC",
                (mission_id,),
            ).fetchall()
            edges = [
                {"from": r["parent_node_id"], "to": r["child_node_id"]}
                for r in conn.execute(
                    "SELECT parent_node_id, child_node_id FROM dag_edges WHERE mission_id = ?",
                    (mission_id,),
                ).fetchall()
            ]

        nodes = [asdict(self.get_node(mission_id, r["node_id"])) for r in n_rows]
        total = max(len(nodes), 1)
        done = sum(1 for n in nodes if n["status"] == "COMPLETED")
        pct = round((done / total) * 100.0, 1)

        mermaid_lines = ["flowchart TD"]
        for n in nodes:
            mermaid_lines.append(f'    {n["node_id"]}["{n["node_id"]}: {n["title"]} ({n["status"]})"]')
        for e in edges:
            mermaid_lines.append(f'    {e["from"]} --> {e["to"]}')

        return {
            "mission_id": mission_id,
            "goal": m_row["goal"] if m_row else "",
            "progress_pct": pct,
            "completed_nodes": done,
            "total_nodes": len(nodes),
            "nodes": nodes,
            "edges": edges,
            "mermaid": "\n".join(mermaid_lines),
        }


def main():
    test_db = ROOT / "outputs" / "test_feral_dag.sqlite"
    for ext in ("", "-wal", "-shm"):
        p = Path(str(test_db) + ext)
        if p.exists():
            p.unlink()

    print("1. Initializing SQLite Hierarchical Task DAG (`DAGOrchestrator`)...")
    dag = DAGOrchestrator(db_path=test_db)
    mission_id = "mission_overnight_refactor"
    dag.create_mission(
        mission_id=mission_id,
        goal="Map repository, patch billing & auth modules in parallel, and run integration test suite.",
    )

    # Create a 4-node Diamond DAG: N1_map -> (N2_billing, N3_auth) -> N4_verify
    dag.add_node(mission_id, "N1_map", "AST Repo Map", "Map all Python modules in src/", assigned_agent="Mapper-01", priority=1)
    dag.add_node(mission_id, "N2_billing", "Patch Billing", "Fix discount formula in billing.py", assigned_agent="Coder-01", depends_on=["N1_map"], priority=2)
    dag.add_node(mission_id, "N3_auth", "Patch Auth Token", "Fix token expiry in auth.py", assigned_agent="Coder-02", depends_on=["N1_map"], priority=3, max_retries=2)
    dag.add_node(mission_id, "N4_verify", "Run Test Suite", "Execute unittest suite across all modules", assigned_agent="Tester-01", depends_on=["N2_billing", "N3_auth"], priority=4)

    snap0 = dag.get_dag_snapshot(mission_id)
    initial_statuses = {n["node_id"]: n["status"] for n in snap0["nodes"]}
    print(f"   - Initial Diamond DAG States : {initial_statuses}")
    assert initial_statuses == {"N1_map": "READY", "N2_billing": "PENDING", "N3_auth": "PENDING", "N4_verify": "PENDING"}

    print("\n2. Testing Cycle Detection Guard (`N4_verify -> N1_map` must be rejected)...")
    cycle_caught = False
    try:
        dag.add_dependency(mission_id, parent_node_id="N4_verify", child_node_id="N1_map")
    except ValueError as e:
        cycle_caught = True
        print(f"   - Cycle Guard Triggered      : {e}")
    assert cycle_caught

    print("\n3. Executing `N1_map` & Verifying Parallel Readiness Promotion (`N2_billing` & `N3_auth`)...")
    n1 = dag.acquire_next_ready_node(mission_id)
    assert n1 is not None and n1.node_id == "N1_map" and n1.status == "RUNNING"
    dag.mark_node_completed(mission_id, "N1_map", "Mapped 2 modules: billing.py and auth.py")

    snap1 = dag.get_dag_snapshot(mission_id)
    after_n1 = {n["node_id"]: n["status"] for n in snap1["nodes"]}
    print(f"   - Post-`N1_map` States       : {after_n1} (Progress: {snap1['progress_pct']}%)")
    assert after_n1 == {"N1_map": "COMPLETED", "N2_billing": "READY", "N3_auth": "READY", "N4_verify": "PENDING"}

    print("\n4. Simulating Process Crash During `N2_billing` & Testing SQLite Restart Recovery...")
    n2 = dag.acquire_next_ready_node(mission_id)
    assert n2 is not None and n2.node_id == "N2_billing" and n2.status == "RUNNING"
    # Re-open a fresh DAGOrchestrator instance pointing at the same SQLite file
    dag_restarted = DAGOrchestrator(db_path=test_db)
    recovered_count = dag_restarted.recover_interrupted_nodes(mission_id)
    print(f"   - Orphaned RUNNING Nodes Recovered on Restart: {recovered_count} (`N2_billing` reset to READY, `N1_map` kept COMPLETED)")
    assert recovered_count == 1
    n2_retry = dag_restarted.acquire_next_ready_node(mission_id)
    assert n2_retry is not None and n2_retry.node_id == "N2_billing"
    dag_restarted.mark_node_completed(mission_id, "N2_billing", "Patched billing.py [patch_id=1]")

    print("\n5. Testing Strike-Counted Self-Healing Retry (`N3_auth` fails Attempt 1 -> retries Attempt 2 -> unlocks `N4_verify`)...")
    n3_try1 = dag_restarted.acquire_next_ready_node(mission_id)
    assert n3_try1 is not None and n3_try1.node_id == "N3_auth" and n3_try1.attempts == 1
    n3_after_fail = dag_restarted.mark_node_failed(mission_id, "N3_auth", "SEARCH block indentation mismatch")
    print(f"   - After Strike 1/{n3_after_fail.max_retries} on `N3_auth`: status={n3_after_fail.status} (auto-requeued)")
    assert n3_after_fail.status == "READY"

    n3_try2 = dag_restarted.acquire_next_ready_node(mission_id)
    assert n3_try2 is not None and n3_try2.node_id == "N3_auth" and n3_try2.attempts == 2
    dag_restarted.mark_node_completed(mission_id, "N3_auth", "Patched auth.py [patch_id=2]")

    # Now both N2_billing and N3_auth are COMPLETED -> N4_verify must automatically promote to READY!
    n4 = dag_restarted.acquire_next_ready_node(mission_id)
    assert n4 is not None and n4.node_id == "N4_verify"
    upstream_ctx = dag_restarted.get_upstream_context(mission_id, "N4_verify")
    print("   - `N4_verify` Automatically Promoted to READY with Upstream Context:")
    for line in upstream_ctx.splitlines():
        print(f"     | {line}")
    dag_restarted.mark_node_completed(mission_id, "N4_verify", "Ran 6 unit tests: OK")

    final_snap = dag_restarted.get_dag_snapshot(mission_id)
    print(f"\n6. Final Mission Snapshot: {final_snap['completed_nodes']}/{final_snap['total_nodes']} nodes COMPLETED ({final_snap['progress_pct']}%)")

    # Clean up test SQLite files
    for ext in ("", "-wal", "-shm"):
        p = Path(str(test_db) + ext)
        if p.exists():
            p.unlink()

    print("\nPHASE 6 STEP 1 (`dag_orchestrator.py`): PASS -> SQLite Task DAG, cycle guard, crash recovery, and strike retry verified!")


if __name__ == "__main__":
    main()
