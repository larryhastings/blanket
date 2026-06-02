#!/usr/bin/env python3

_license = """
blanket
Copyright 2025-2026 Larry Hastings
All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR
THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import importlib
import io
import pathlib
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from unittest import mock

import blankettestlib
blankettestlib.preload_local_blanket()

import test_all


class TestBlanketTestlib(unittest.TestCase):
    def test_preload_local_blanket_fails_cleanly_at_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv_0 = pathlib.Path(tmp) / "nowhere" / "runner.py"
            with mock.patch.object(sys, "argv", [str(argv_0)]):
                with self.assertRaises(FileNotFoundError) as cm:
                    blankettestlib.preload_local_blanket()
        self.assertIn("could not find local blanket package", str(cm.exception))

    def test_run_accumulates_failures_and_prints_summary(self):
        class DeliberatelyFailing(unittest.TestCase):
            def test_failure(self):
                self.fail("intentional failure")

        module = types.ModuleType("deliberately_failing_tests")
        module.DeliberatelyFailing = DeliberatelyFailing
        module.__dict__["__unittest"] = True
        old_stats = blankettestlib.stats.copy()
        try:
            blankettestlib.stats.update({name: 0 for name in blankettestlib.stats})
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                blankettestlib.run("deliberate", module)
                blankettestlib.finish()
            output = stdout.getvalue()
        finally:
            blankettestlib.stats.update(old_stats)
        self.assertIn("Testing deliberate...", output)
        self.assertIn("FAILED (failures=1)", output)

    def test_run_prints_permutation_count(self):
        class Passing(unittest.TestCase):
            def test_pass(self):
                self.assertTrue(True)

        module = types.ModuleType("passing_tests")
        module.Passing = Passing
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            blankettestlib.run("permutations", module, permutations=lambda: 37)
        self.assertIn("1 test, with 37 total permutations", stdout.getvalue())

    def test_run_without_name_accepts_module_object(self):
        class Passing(unittest.TestCase):
            def test_pass(self):
                self.assertTrue(True)

        module = types.ModuleType("passing_object_tests")
        module.Passing = Passing
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            blankettestlib.run(None, module)
        self.assertIn("Ran 1 test", stdout.getvalue())

    def test_run_without_name_accepts_module_string(self):
        class Passing(unittest.TestCase):
            def test_pass(self):
                self.assertTrue(True)

        module = types.ModuleType("passing_string_tests")
        module.Passing = Passing
        sys.modules[module.__name__] = module
        try:
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                blankettestlib.run(None, module.__name__)
        finally:
            sys.modules.pop(module.__name__, None)
        self.assertIn("Ran 1 test", stdout.getvalue())

    def test_preload_local_blanket_handles_existing_sys_path_entry(self):
        root = pathlib.Path(__file__).resolve().parents[1]
        old_path = list(sys.path)
        try:
            sys.path[:] = [root] + [p for p in sys.path if p != root]
            with mock.patch.object(sys, "argv", [str(root / "tests" / "test_harness.py")]):
                found = blankettestlib.preload_local_blanket()
            self.assertEqual(found, root)
            self.assertIs(sys.path[0], root)
        finally:
            sys.path[:] = old_path

    def test_run_prints_output_without_ran_line(self):
        class Result:
            failures = ()
            errors = ()
            expectedFailures = ()
            unexpectedSuccesses = ()
            skipped = ()

        class MainResult:
            result = Result()

        with mock.patch.object(blankettestlib.unittest, "main", return_value=MainResult()):
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                blankettestlib.run("quiet", types.ModuleType("quiet_tests"))
        self.assertEqual(stdout.getvalue(), "Testing quiet...\n\n")


class TestTestAll(unittest.TestCase):
    def test_test_modules_contains_base_and_current_runtime_modules(self):
        modules = test_all.test_modules()
        self.assertGreaterEqual(
            modules.index("test_primitives"),
            modules.index("test_injector"),
        )
        for name in ("test_release_metadata", "test_stdlib_fidelity", "test_injector", "test_primitives", "test_harness"):
            with self.subTest(name=name):
                self.assertIn(name, modules)
        self.assertEqual(len(modules), len(set(modules)))

    def test_run_modules_imports_and_runs_in_order(self):
        calls = []
        fake_modules = {}
        for name in ("alpha", "beta"):
            module = types.ModuleType(name)
            module.run_tests = lambda name=name: calls.append(name)
            fake_modules[name] = module

        real_module = types.ModuleType("real_runner_between_fakes")
        real_module.run_tests = lambda: calls.append("real")
        sys.modules[real_module.__name__] = real_module
        real_import = __import__
        def fake_import(name, *args, **kwargs):
            if name in fake_modules:
                return fake_modules[name]
            return real_import(name, *args, **kwargs)

        try:
            with mock.patch("builtins.__import__", fake_import):
                test_all.run_modules(["alpha", real_module.__name__, "beta"])
        finally:
            sys.modules.pop(real_module.__name__, None)
        self.assertEqual(calls, ["alpha", "real", "beta"])

    def test_run_modules_falls_through_to_real_import(self):
        calls = []
        module = types.ModuleType("real_imported_runner")
        module.run_tests = lambda: calls.append("real")
        sys.modules[module.__name__] = module
        try:
            test_all.run_modules([module.__name__])
        finally:
            sys.modules.pop(module.__name__, None)
        self.assertEqual(calls, ["real"])

    def test_test_module_run_tests_functions_delegate_to_blankettestlib_run(self):
        module_names = test_all.test_modules()
        calls = []
        def fake_run(*args, **kwargs):
            calls.append((args, kwargs))

        with mock.patch.object(blankettestlib, "run", fake_run):
            for module_name in module_names:
                module = importlib.import_module(module_name)
                module.run_tests()

        self.assertEqual(len(calls), len(module_names))
        self.assertTrue(all(args or kwargs for args, kwargs in calls))

    def test_main_preloads_runs_and_finishes(self):
        calls = []
        with mock.patch.object(test_all.blankettestlib, "preload_local_blanket", lambda: calls.append("preload")):
            with mock.patch.object(test_all, "test_modules", lambda: ["fake"]):
                with mock.patch.object(test_all, "run_modules", lambda modules: calls.append(("run", tuple(modules)))):
                    with mock.patch.object(test_all.blankettestlib, "finish", lambda: calls.append("finish")):
                        test_all.main()
        self.assertEqual(calls, ["preload", ("run", ("fake",)), "finish"])

    def test_test_all_can_be_imported_twice(self):
        self.assertIs(importlib.import_module("test_all"), test_all)


def run_tests():
    blankettestlib.run(name="blanket.harness", module=__name__)


if __name__ == '__main__':
    run_tests()
    blankettestlib.finish()
