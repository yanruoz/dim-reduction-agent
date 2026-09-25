"""Deterministic preflight check, mirrors validate_plan.py's pass/fail +
   JSON+Markdown report pattern. Scoped per-activity (core vs. report) so a
   report-only gap never blocks profiling through evaluation, per
   int-brain-lab/ibl-ai-agent's install skill (docs/design.md §1 item 8).
"""
import argparse
import importlib
import json
import sys
from pathlib import Path

import loaders

CORE_PACKAGES = ["numpy", "pandas", "scipy", "sklearn", "anndata", "umap", "matplotlib"]
REPORT_PACKAGES = ["matplotlib.backends.backend_pdf"]


def _check_imports(module_names):
    results = []
    for name in module_names:
        try:
            importlib.import_module(name)
            results.append({"module": name, "ok": True})
        except ImportError as e:
            results.append({"module": name, "ok": False, "error": str(e)})
    return results


def check_core(dataset=None):
    checks = _check_imports(CORE_PACKAGES)
    errors = [f"Cannot import '{c['module']}': {c['error']}" for c in checks if not c["ok"]]

    if dataset:
        folder = loaders.DATA_ROOT / dataset
        desc_path = folder / "DATA_DESCRIPTION.md"
        exists = desc_path.exists()
        checks.append({"module": f"data/{dataset}/DATA_DESCRIPTION.md", "ok": exists})
        if not exists:
            errors.append(f"Missing {desc_path}")
        else:
            # The data file itself: resolvable per the '## Loading' section (or the single
            # data file in the folder). If missing but a url is declared, say how to get it.
            spec = {}
            try:
                spec = loaders.parse_loading_spec(desc_path)
                data_file = loaders._find_data_file(folder, spec)
                checks.append({"module": f"data/{dataset}/{data_file.name}", "ok": True})
            except ValueError as e:
                hint = ""
                if "url" in spec:
                    hint = f" A url is declared: run `python scripts/fetch_data.py --dataset {dataset}`."
                checks.append({"module": f"data file for {dataset}", "ok": False})
                errors.append(f"{e}{hint}")

    return errors, checks


def check_report():
    checks = _check_imports(REPORT_PACKAGES)
    errors = [f"Cannot import '{c['module']}': {c['error']}" for c in checks if not c["ok"]]
    return errors, checks


def write_reports(errors, checks, check_name):
    report = {"check": check_name, "passed": not errors, "errors": errors, "checks": checks}
    json_path = Path("doctor_report.json")
    md_path = Path("doctor_report.md")
    json_path.write_text(json.dumps(report, indent=2) + "\n")

    lines = [f"# Doctor check: {check_name}", "", f"**Result:** {'PASS' if not errors else 'FAIL'}", "", "## Checks"]
    lines += [f"- {'OK' if c['ok'] else 'FAIL'}: {c['module']}" for c in checks]
    if errors:
        lines += ["", "## Errors"] + [f"- {e}" for e in errors]
    md_path.write_text("\n".join(lines) + "\n")
    return json_path, md_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deterministic preflight check for required dependencies and inputs.")
    parser.add_argument("--check", choices=["core", "report", "all"], default="all")
    parser.add_argument("--dataset", default=None, help="If given, also checks data/<dataset>/DATA_DESCRIPTION.md exists (core check only)")
    args = parser.parse_args()

    all_errors, all_checks = [], []
    if args.check in ("core", "all"):
        e, c = check_core(args.dataset)
        all_errors += e
        all_checks += c
    if args.check in ("report", "all"):
        e, c = check_report()
        all_errors += e
        all_checks += c

    json_path, md_path = write_reports(all_errors, all_checks, args.check)

    print(f"{'PASS' if not all_errors else 'FAIL'}: doctor --check {args.check}")
    for c in all_checks:
        print(f"  {'OK' if c['ok'] else 'FAIL'}: {c['module']}")
    for e in all_errors:
        print(f"  ERROR: {e}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")

    sys.exit(0 if not all_errors else 1)
