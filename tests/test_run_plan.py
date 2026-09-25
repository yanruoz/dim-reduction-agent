"""Tests for the clustering step and figure-freshness logic in scripts/run_plan.py.
The script runner is replaced by a fake so no real reduction runs.

Run from the repo root:  venv/bin/python -m unittest discover -s tests -v
"""
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import run_plan  # noqa: E402


def fake_result(returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess([], returncode, stdout=stdout, stderr=stderr)


class RunPlanCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._cwd = os.getcwd()
        os.chdir(self._tmp.name)
        self.out = Path("outputs/d")
        for sub in ("embeddings", "metrics", "figures"):
            (self.out / sub).mkdir(parents=True)
        self.calls = []
        self._orig = run_plan._run_script
        run_plan._run_script = self.fake_run

    def tearDown(self):
        run_plan._run_script = self._orig
        os.chdir(self._cwd)
        self._tmp.cleanup()

    responses = {}

    def fake_run(self, script, args):
        self.calls.append((script, args))
        return self.responses.get(script, fake_result())

    def make_clusters(self, params=None):
        (self.out / "clusters.npy").write_bytes(b"x")
        (self.out / "metrics" / "clustering.json").write_text(json.dumps({"params": params or {"source": "pca", "k": None}}))

    def make_method(self, name, figure=True):
        (self.out / "embeddings" / f"{name}.npy").write_bytes(b"x")
        if figure:
            time.sleep(0.01)
            (self.out / "figures" / f"{name}.png").write_bytes(b"x")


class TestRunClustering(RunPlanCase):
    plan = {"clustering": {"source": "pca", "reason": "r"}}
    ok_logs = [{"name": "pca", "status": "ok"}, {"name": "umap", "status": "ok"}]

    def test_no_block_means_no_call_and_leftover_files_are_removed(self):
        self.make_clusters()
        self.assertIsNone(run_plan.run_clustering("d", {}, self.ok_logs))
        self.assertEqual(self.calls, [])
        self.assertFalse((self.out / "clusters.npy").exists())
        self.assertFalse((self.out / "metrics" / "clustering.json").exists())

    def test_success_calls_cluster_py_and_reports_its_status(self):
        self.responses = {"cluster.py": fake_result(stdout="clustering: reused\n")}
        entry = run_plan.run_clustering("d", self.plan, self.ok_logs)
        self.assertEqual(self.calls, [("cluster.py", ["--dataset", "d"])])
        self.assertEqual(entry, {"status": "reused"})

    def test_a_reused_source_method_is_fine(self):
        self.responses = {"cluster.py": fake_result(stdout="clustering: ok\n")}
        entry = run_plan.run_clustering("d", self.plan, [{"name": "pca", "status": "reused"}])
        self.assertEqual(entry["status"], "ok")

    def test_failure_is_not_fatal_clears_stale_files_and_is_logged(self):
        self.make_clusters()
        self.responses = {"cluster.py": fake_result(1, stderr="Traceback\nValueError: boom")}
        entry = run_plan.run_clustering("d", self.plan, self.ok_logs)
        self.assertEqual(entry["status"], "failed")
        self.assertIn("boom", entry["error"])
        self.assertFalse((self.out / "clusters.npy").exists())

    def test_a_failed_source_method_skips_clustering_and_clears_stale_files(self):
        self.make_clusters()
        logs = [{"name": "pca", "status": "fell_back_to_pca"}]
        entry = run_plan.run_clustering("d", self.plan, logs)
        self.assertEqual(entry["status"], "skipped")
        self.assertEqual(self.calls, [])
        self.assertFalse((self.out / "clusters.npy").exists())

    def test_a_source_that_is_not_in_the_run_is_skipped(self):
        entry = run_plan.run_clustering("d", {"clustering": {"source": "kernel_pca"}}, self.ok_logs)
        self.assertEqual(entry["status"], "skipped")


class TestColoringKey(RunPlanCase):
    def test_default_without_clusters(self):
        self.assertEqual(run_plan.coloring_key("d"), "default")

    def test_key_changes_when_cluster_params_change_and_is_stable_otherwise(self):
        self.make_clusters({"source": "pca", "k": None})
        a = run_plan.coloring_key("d")
        self.assertEqual(a, run_plan.coloring_key("d"))
        self.make_clusters({"source": "pca", "k": 4})
        self.assertNotEqual(a, run_plan.coloring_key("d"))
        self.assertTrue(a.startswith("clusters:"))


class TestFigureIsCurrent(RunPlanCase):
    def stamp(self, key):
        (self.out / "figures" / "coloring.json").write_text(json.dumps({"key": key}))

    def test_current_when_figure_is_newer_than_embedding_and_key_matches(self):
        (self.out / "embeddings" / "pca.npy").write_bytes(b"x")
        time.sleep(0.01)
        (self.out / "figures" / "pca.png").write_bytes(b"x")
        self.stamp("default")
        self.assertTrue(run_plan.figure_is_current("d", "pca", "default"))

    def test_missing_figure_is_not_current(self):
        (self.out / "embeddings" / "pca.npy").write_bytes(b"x")
        self.stamp("default")
        self.assertFalse(run_plan.figure_is_current("d", "pca", "default"))

    def test_embedding_newer_than_figure_is_not_current(self):
        self.make_method("pca")
        time.sleep(0.01)
        (self.out / "embeddings" / "pca.npy").write_bytes(b"y")
        self.stamp("default")
        self.assertFalse(run_plan.figure_is_current("d", "pca", "default"))

    def test_changed_coloring_key_or_no_marker_is_not_current(self):
        self.make_method("pca")
        self.assertFalse(run_plan.figure_is_current("d", "pca", "default"))  # no marker yet
        self.stamp("clusters:abc")
        self.assertFalse(run_plan.figure_is_current("d", "pca", "default"))
        self.assertTrue(run_plan.figure_is_current("d", "pca", "clusters:abc"))


class TestRenderFigures(RunPlanCase):
    def test_only_successful_methods_are_drawn(self):
        self.make_method("pca", figure=False)
        self.make_method("umap", figure=False)
        logs = [{"name": "pca", "status": "ok"}, {"name": "umap", "status": "fell_back_to_pca"}, {"name": "mds", "status": "reused"}]
        self.make_method("mds", figure=False)
        run_plan.render_figures("d", logs)
        drawn = [a[a.index("--method") + 1] for s, a in self.calls if s == "visualize.py"]
        self.assertEqual(sorted(drawn), ["mds", "pca"])

    def test_current_figures_are_not_redrawn_and_stale_ones_are(self):
        self.make_method("pca")
        (self.out / "figures" / "coloring.json").write_text(json.dumps({"key": "default"}))
        logs = [{"name": "pca", "status": "reused"}]
        run_plan.render_figures("d", logs)
        self.assertEqual(self.calls, [])
        self.make_clusters()  # coloring changes to clusters
        run_plan.render_figures("d", logs)
        self.assertEqual([s for s, _ in self.calls], ["visualize.py"])
        key = json.loads((self.out / "figures" / "coloring.json").read_text())["key"]
        self.assertEqual(key, run_plan.coloring_key("d"))

    def test_a_failed_figure_is_logged_and_the_coloring_is_not_stamped(self):
        self.make_method("pca", figure=False)
        self.responses = {"visualize.py": fake_result(1, stderr="RuntimeError: bad plot")}
        logs = [{"name": "pca", "status": "ok"}]
        run_plan.render_figures("d", logs)
        self.assertEqual(logs[0]["status"], "visualize_failed")
        self.assertIn("bad plot", logs[0]["error"])
        self.assertFalse((self.out / "figures" / "coloring.json").exists())


if __name__ == "__main__":
    unittest.main()
