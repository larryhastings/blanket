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

from pathlib import Path
import re
import unittest

import blankettestlib
blankettestlib.preload_local_blanket()

import blanket


ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"


class TestReleaseMetadata(unittest.TestCase):
    """Small checks that keep the release metadata from drifting."""

    @classmethod
    def setUpClass(cls):
        cls.pyproject = PYPROJECT.read_text(encoding="utf-8")

    def test_version_is_1_1(self):
        self.assertEqual(blanket.__version__, "1.1")

    def test_python_requirement_and_classifiers(self):
        self.assertIn('requires-python = ">=3.10"', self.pyproject)
        for version in ("3.10", "3.11", "3.12", "3.13", "3.14"):
            with self.subTest(version=version):
                self.assertIn(
                    f'"Programming Language :: Python :: {version}"',
                    self.pyproject,
                )
        self.assertIn(
            '"Programming Language :: Python :: Implementation :: CPython"',
            self.pyproject,
        )

    def test_build_backend_and_runtime_dependencies(self):
        self.assertIn('build-backend = "flit_core.buildapi"', self.pyproject)
        for dependency in ("bytecode >= 0.17", "big >= 0.13.2"):
            with self.subTest(dependency=dependency):
                pattern = r'"' + re.escape(dependency) + r'"'
                self.assertRegex(self.pyproject, pattern)

    def test_supported_python_coverage_configs_exist(self):
        for version in ("310", "311", "312", "313", "314"):
            with self.subTest(version=version):
                path = ROOT / f".coveragerc.py{version}"
                self.assertTrue(path.exists(), path)
                contents = path.read_text(encoding="utf-8")
                self.assertIn("source =", contents)
                self.assertIn("    blanket", contents)
                self.assertIn("    tests", contents)
                self.assertIn("fail_under = 100", contents)

    def test_coverage_configs_omit_version_inapplicable_test_files(self):
        expected_omits = {
            "310": (
                "tests/test_injector_py311_plus.py",
                "tests/test_primitives_py313_plus.py",
            ),
            "311": ("tests/test_primitives_py313_plus.py",),
            "312": ("tests/test_primitives_py313_plus.py",),
            "313": (),
            "314": (),
        }
        for version, omitted_files in expected_omits.items():
            with self.subTest(version=version):
                contents = (ROOT / f".coveragerc.py{version}").read_text(encoding="utf-8")
                for omitted_file in omitted_files:
                    self.assertIn(f"    {omitted_file}", contents)
                if not omitted_files:
                    self.assertNotIn("omit =", contents)

    def test_no_uv_dependency(self):
        # blanket should build and test with the usual Python tools.  uv is
        # fine as a contributor's local convenience, but it isn't a project
        # dependency.
        self.assertNotRegex(self.pyproject, r'(?i)\buv\b')


def run_tests():
    blankettestlib.run(name="blanket.release_metadata", module=__name__)


if __name__ == '__main__':
    run_tests()
    blankettestlib.finish()
