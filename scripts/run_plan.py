"""Orchestrates reduce_dim.py -> evaluate.py for every method already in a
   validated plan.json (never adds to that list), then cluster.py if the plan
   has a `clustering` block, then visualize.py for every method that
   succeeded (figures come last because unlabeled plots are colored by the
   clusters). Skips steps when matching artifacts already exist (CLAUDE.md
   rule 13, reuse before recomputing); retries once on a likely disconnected-
   neighbor-graph failure for the three methods known to hit that; falls back
   to noting PCA in that slot if a method still fails, rather than aborting
   the whole run. Writes outputs/<dataset>/run_log.json.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

RETRY_ELIGIBLE_METHODS = {"isomap", "lle", "laplacian_eigenmaps"}
DISCONNECTED_KEYWORDS = ("disconnected", "not fully connected", "connected component")


def _run_script(script, args_list):
    cmd = [sys.executable, str(Path(__file__).parent / script)] + args_list
    return subprocess.run(cmd, capture_output=True, text=True)


def _last_error_line(result):
    lines = result.stderr.strip().splitlines()
    return lines[-1] if lines else "unknown error"


def _embedding_matches_plan(dataset, method_entry, seed):
    embedding_path = Path("outputs") / dataset / "embeddings" / f"{method_entry['name']}.npy"
    sidecar_path = embedding_path.with_suffix(".json")
    if not embedding_path.exists() or not sidecar_path.exists():
        return False
    sidecar = json.loads(sidecar_path.read_text())
    same_hp = json.dumps(sidecar.get("hyperparameters", {}), sort_keys=True) == json.dumps(
        method_entry.get("hyperparameters", {}), sort_keys=True
    )
    return same_hp and sidecar.get("seed") == seed


def run_method(dataset, method_entry, seed, index, total):
    name = method_entry["name"]
    print(f"[{index}/{total}] Running {name}...", flush=True)
    log_entry = {"name": name}

    metrics_path = Path("outputs") / dataset / "metrics" / f"{name}.json"

    embedding_ok = _embedding_matches_plan(dataset, method_entry, seed)
    if embedding_ok and metrics_path.exists():
        print("  reused (matching artifacts already exist)")
        log_entry["status"] = "reused"
        return log_entry

    params_json = json.dumps(method_entry.get("hyperparameters", {}))

    if not embedding_ok:
        start = time.time()
        result = _run_script(
            "reduce_dim.py", ["--dataset", dataset, "--method", name, "--params", params_json, "--seed", str(seed)]
        )
        if result.returncode != 0:
            disconnected = any(kw in result.stderr.lower() for kw in DISCONNECTED_KEYWORDS)
            if name in RETRY_ELIGIBLE_METHODS and disconnected:
                print("  neighbor graph disconnected, retrying with a larger n_neighbors...")
                retry_hp = dict(method_entry.get("hyperparameters", {}))
                retry_hp["n_neighbors"] = int(retry_hp.get("n_neighbors", 10) * 2)
                log_entry["retried"] = True
                result = _run_script(
                    "reduce_dim.py",
                    ["--dataset", dataset, "--method", name, "--params", json.dumps(retry_hp), "--seed", str(seed)],
                )

            if result.returncode != 0:
                print("  FAILED, falling back to PCA for this slot")
                log_entry["status"] = "fell_back_to_pca"
                log_entry["error"] = _last_error_line(result)
                return log_entry
        print(f"  reduce_dim done ({time.time() - start:.1f}s)")

    if not embedding_ok or not metrics_path.exists():
        result = _run_script("evaluate.py", ["--dataset", dataset, "--method", name])
        if result.returncode != 0:
            log_entry["status"] = "evaluate_failed"
            log_entry["error"] = _last_error_line(result)
            return log_entry

    log_entry["status"] = "ok"
    return log_entry


def _clusters_paths(dataset):
    out = Path("outputs") / dataset
    return out / "clusters.npy", out / "metrics" / "clustering.json"


def run_clustering(dataset, plan, method_logs):
    """Runs cluster.py when the plan asks for it. Never fatal: on any problem the stale cluster files are
    removed so figures fall back to density coloring, and the reason is logged. Returns a run_log entry
    (or None when the plan has no clustering block, after clearing leftovers from an earlier plan)."""
    labels_path, metrics_path = _clusters_paths(dataset)

    def clear():
        for path in (labels_path, metrics_path):
            path.unlink(missing_ok=True)

    block = plan.get("clustering")
    if block is None:
        clear()
        return None

    print(f"Clustering ({block.get('algorithm', 'kmeans')} on {block.get('source')}), used only to color plots...", flush=True)
    source_status = next((m["status"] for m in method_logs if m["name"] == block.get("source")), None)
    if source_status not in ("ok", "reused"):
        clear()
        print("  skipped: the source method did not succeed; unlabeled plots fall back to density coloring")
        return {"status": "skipped", "error": f"source method {block.get('source')!r} status: {source_status}"}

    result = _run_script("cluster.py", ["--dataset", dataset])
    if result.returncode != 0:
        clear()
        print("  FAILED; unlabeled plots fall back to density coloring")
        return {"status": "failed", "error": _last_error_line(result)}
    status = result.stdout.strip().split(":")[-1].strip() or "ok"
    print(f"  {status}")
    return {"status": status}


def coloring_key(dataset):
    """Identifies what unlabeled figures are colored by, so figures are redrawn when it changes."""
    _, metrics_path = _clusters_paths(dataset)
    if not metrics_path.exists():
        return "default"
    params = json.loads(metrics_path.read_text()).get("params", {})
    return "clusters:" + hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]


def figure_is_current(dataset, name, key):
    out = Path("outputs") / dataset
    figure = out / "figures" / f"{name}.png"
    embedding = out / "embeddings" / f"{name}.npy"
    marker = out / "figures" / "coloring.json"
    if not figure.exists() or figure.stat().st_mtime < embedding.stat().st_mtime:
        return False
    return marker.exists() and json.loads(marker.read_text()).get("key") == key


def render_figures(dataset, method_logs):
    """visualize.py for every method that produced an embedding and metrics, unless its figure is current."""
    key = coloring_key(dataset)
    figures_dir = Path("outputs") / dataset / "figures"
    for entry in method_logs:
        if entry["status"] not in ("ok", "reused"):
            continue
        if figure_is_current(dataset, entry["name"], key):
            continue
        result = _run_script("visualize.py", ["--dataset", dataset, "--method", entry["name"]])
        if result.returncode != 0:
            entry["status"] = "visualize_failed"
            entry["error"] = _last_error_line(result)
    figures_dir.mkdir(parents=True, exist_ok=True)
    # Only stamp the coloring once every figure that should exist has been drawn with it
    if all(e["status"] != "visualize_failed" for e in method_logs):
        (figures_dir / "coloring.json").write_text(json.dumps({"key": key}) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run every method in a validated plan.json: reduce_dim -> evaluate -> visualize.")
    parser.add_argument("plan", help="Path to plan.json (outputs/<dataset>/plan.json convention)")
    args = parser.parse_args()

    plan_path = Path(args.plan)
    dataset = plan_path.parent.name  # authoritative over plan["dataset"], matches the actual file location
    plan = json.loads(plan_path.read_text())
    seed = plan.get("seed", 0)
    methods = plan["methods"]

    log = {"dataset": dataset, "plan_revision": plan.get("revision"), "methods": []}
    for i, method_entry in enumerate(methods, start=1):
        log["methods"].append(run_method(dataset, method_entry, seed, i, len(methods)))

    clustering_log = run_clustering(dataset, plan, log["methods"])
    if clustering_log is not None:
        log["clustering"] = clustering_log

    print("Rendering figures...", flush=True)
    render_figures(dataset, log["methods"])

    log_path = Path("outputs") / dataset / "run_log.json"
    log_path.write_text(json.dumps(log, indent=2) + "\n")
    print(f"Wrote {log_path}")

    n_needed_fallback = sum(1 for m in log["methods"] if m["status"] not in ("ok", "reused"))
    if n_needed_fallback:
        print(f"{n_needed_fallback} of {len(methods)} methods needed a fallback; see {log_path}")
