import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "parameters with range - advance/run.py"


class ExampleTests(unittest.TestCase):
    def test_import_does_not_run_search_or_read_yaml(self):
        spec = importlib.util.spec_from_file_location("example", EXAMPLE)
        module = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(EXAMPLE.parent))
        try:
            with contextlib.redirect_stdout(io.StringIO()) as output:
                spec.loader.exec_module(module)
            self.assertEqual(output.getvalue(), "")
        finally:
            sys.path.pop(0)

    @unittest.skipUnless(importlib.util.find_spec("yaml"), "optional examples dependency not installed")
    def test_portable_cli_reproduces_and_uses_all_configured_genes(self):
        command = [sys.executable, str(EXAMPLE), "--seed", "7", "--generations", "2",
                   "--population-size", "5"]
        with tempfile.TemporaryDirectory() as directory:
            first = json.loads(subprocess.check_output(command, cwd=directory, text=True))
            second = json.loads(subprocess.check_output(command, cwd=directory, text=True))
        self.assertEqual(first, second)
        self.assertEqual(first["evaluations"], 15)
        self.assertEqual(len(first["best_individual"]), 28)
        self.assertAlmostEqual(first["best_fitness"], sum(first["best_individual"].values()))


if __name__ == "__main__":
    unittest.main()
