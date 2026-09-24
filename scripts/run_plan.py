"""Orchestrates reduce_dim.py -> evaluate.py -> visualize.py for every method
   already in a validated plan.json (never adds to that list). Skips a
   method's steps when matching artifacts already exist (CLAUDE.md rule 13,
   reuse before recomputing); retries once on a likely disconnected-neighbor-
   graph failure for the three methods known to hit that; falls back to
   noting PCA in that slot if a method still fails, rather than aborting the
   whole run. Writes outputs/<dataset>/run_log.json.
"""
import argparse
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

    embedding_path = Path("outputs") / dataset / "embeddings" / f"{name}.npy"
    metrics_path = Path("outputs") / dataset / "metrics" / f"{name}.json"
    figure_path = Path("outputs") / dataset / "figures" / f"{name}.png"

    embedding_ok = _embedding_matches_plan(dataset, method_entry, seed)
    if embedding_ok and metrics_path.exists() and figure_path.exists():
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

    if not embedding_ok or not figure_path.exists():
        result = _run_script("visualize.py", ["--dataset", dataset, "--method", name])
        if result.returncode != 0:
            log_entry["status"] = "visualize_failed"
            log_entry["error"] = _last_error_line(result)
            return log_entry

    log_entry["status"] = "ok"
    return log_entry


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

    log_path = Path("outputs") / dataset / "run_log.json"
    log_path.write_text(json.dumps(log, indent=2) + "\n")
    print(f"Wrote {log_path}")

    n_needed_fallback = sum(1 for m in log["methods"] if m["status"] not in ("ok", "reused"))
    if n_needed_fallback:
        print(f"{n_needed_fallback} of {len(methods)} methods needed a fallback; see {log_path}")
