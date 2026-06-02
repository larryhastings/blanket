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

import contextlib
import inspect
import io
import pathlib
import sys
import unittest
import time
import threading
import tokenize
import tempfile

import blankettestlib
blankettestlib.preload_local_blanket()

from blanket import Location, inject_call
import blanket.injector as injector_module
from blanket.injector import _find_statement_end

# Test decorators for version-specific tests
version_info = sys.version_info[0:2]
_python_3_8_plus  = sys.version_info >= (3,  8)
_python_3_11_plus = sys.version_info >= (3, 11)
_python_3_13_plus = sys.version_info >= (3, 13)



def _strip_line_and_column_information(fn, *, firstlineno, name=None, qualname=None):
    """
    Remove line and column information from a function.  Supports Python 3.6-3.14.

    Pass an obviously invalid firstlineno value (like -1) to test corrupt code objects,
    or None to use the original function's co_firstlineno.
    """
    fn_type = type(fn)

    co = fn.__code__
    code_type = type(co)
    if firstlineno is None:
        firstlineno = co.co_firstlineno

    if name is not None:
        co_name = fn_name = name
    else:
        co_name = co.co_name
        fn_name = fn.__name__

    if _python_3_11_plus:
        if qualname is not None:
            co_qualname = qualname
        elif name is not None:
            co_qualname = name
        else:
            co_qualname = co.co_qualname

    code_args = [
        co.co_argcount,
        co.co_kwonlyargcount,
        co.co_nlocals,
        co.co_stacksize,
        co.co_flags,
        co.co_code,
        co.co_consts,
        co.co_names,
        co.co_varnames,
        co.co_filename,
        co_name,
        firstlineno,
        b'', # lnotab in 3.6-3.9, linetable in 3.10+ -- either way, this is the line number information
        co.co_freevars,
        co.co_cellvars,
        ]

    if _python_3_8_plus:
        # added posonlyargcount, inconveniently as positional argument 1 (after argcount)
        code_args.insert(1, co.co_posonlyargcount)

    if _python_3_11_plus:
        # added qualname, inconveniently as positional argument 12 (after name)
        code_args.insert(12, co_qualname)
        # added exceptiontable, inconveniently as positional argument 15 (after linetable)
        code_args.insert(15, co.co_exceptiontable)

    code = code_type(*code_args)

    fn_args = [
        code,
        fn.__globals__,
        fn_name,
        fn.__defaults__,
        fn.__closure__,
        ]

    if _python_3_13_plus:
        fn_args.append(fn.__kwdefaults__)

    fn = fn_type(*fn_args)
    if qualname is not None:
        fn.__qualname__ = qualname
    return fn


# Test code
def sample_function(n=0):
    x = 1
    y = 2
    if n:
        return n * 2
    return x + y


def function_with_trailing_uncompiled_line():
    x = 1
    return x
    # this line is part of inspect.getsource(), but compiles to no bytecode


def do_nothing():
    pass

do_nothing()  # Called for coverage


global_value = 0


def add_100_to_global_value():
    global global_value
    global_value += 100



class TestBlanketTestLib(unittest.TestCase):

    def test_module_helper_functions_execute(self):
        global global_value
        self.assertEqual(function_with_trailing_uncompiled_line(), 1)
        global_value = 0
        add_100_to_global_value()
        self.assertEqual(global_value, 100)

    def test_finish_reports_ok_or_failed_and_stats(self):
        old_stats = blankettestlib.stats.copy()
        try:
            for key in blankettestlib.stats:
                blankettestlib.stats[key] = 0

            sio = io.StringIO()
            with contextlib.redirect_stdout(sio):
                blankettestlib.finish()
            self.assertEqual(sio.getvalue(), "OK\n")

            for key in blankettestlib.stats:
                blankettestlib.stats[key] = 0
            blankettestlib.stats['errors'] = 1
            blankettestlib.stats['skipped'] = 2

            sio = io.StringIO()
            with contextlib.redirect_stdout(sio):
                blankettestlib.finish()
            self.assertEqual(sio.getvalue(), "FAILED (errors=1, skipped=2)\n")
        finally:
            blankettestlib.stats.update(old_stats)

    def test_preload_local_blanket_raises_if_search_hits_root(self):
        old_argv0 = sys.argv[0]
        with tempfile.TemporaryDirectory() as directory:
            try:
                sys.argv[0] = str(pathlib.Path(directory) / "runner.py")
                with self.assertRaises(FileNotFoundError):
                    blankettestlib.preload_local_blanket()
            finally:
                sys.argv[0] = old_argv0


class TestInjectorVersionHelpers(unittest.TestCase):

    def test_export_accepts_explicit_string_names(self):
        original_all = injector_module.__all__.copy()
        try:
            self.assertEqual(injector_module.export('made_up_name'), 'made_up_name')
            self.assertIn('made_up_name', injector_module.__all__)
        finally:
            injector_module.__all__[:] = original_all

    def test_bound_public_helpers_keep_public_names(self):
        """Binding-time helper selection should preserve introspection names."""
        for name in (
            '_instr_line_column',
            '_validate_match_at_line_start',
            '_text_position_to_search_start_position',
            '_find_bytecode_range_for_source_range',
            '_insert_call_bytecode',
            ):
            self.assertEqual(getattr(injector_module, name).__name__, name)

        self.assertEqual(Location.position.__name__, 'position')
        self.assertEqual(Location.position.__qualname__, 'Location.position')
        self.assertEqual(Location._position_py310_minus.__name__, 'position')
        self.assertEqual(Location._position_py310_minus.__qualname__, 'Location.position')
        self.assertEqual(Location._position_py311_plus.__name__, 'position')
        self.assertEqual(Location._position_py311_plus.__qualname__, 'Location.position')

    def test_py310_minus_line_column_helper_runs_on_current_python(self):
        """The old line-only helper is directly testable on modern Python."""
        rows = list(injector_module._instr_line_column_py310_minus(sample_function))
        self.assertTrue(rows)
        self.assertTrue(all(column == 0 for instr, line, column in rows))
        self.assertTrue(all(line >= sample_function.__code__.co_firstlineno for instr, line, column in rows))

    def test_py310_minus_line_column_helper_accepts_old_starts_line_shape(self):
        """The old helper still handles the pre-3.13 starts_line shape."""
        class FakeCode:
            co_firstlineno = 10

        class FakeFunction:
            __code__ = FakeCode()

        class FakeInstruction:
            def __init__(self, starts_line):
                self.starts_line = starts_line

        class FakeDis:
            @staticmethod
            def get_instructions(function):
                return iter((
                    FakeInstruction(None),
                    FakeInstruction(12),
                    FakeInstruction(None),
                    ))

        old_dis = injector_module.dis
        try:
            # Do not mutate the real stdlib dis module here.  Coverage's
            # Python 3.14 sys.monitoring tracer uses dis.get_instructions
            # while this test is running.  Rebind blanket.injector's global
            # instead, so only the helper under test sees the fake shape.
            injector_module.dis = FakeDis
            rows = list(injector_module._instr_line_column_py310_minus(FakeFunction()))
        finally:
            injector_module.dis = old_dis

        self.assertEqual([line for instr, line, column in rows], [10, 12, 12])
        self.assertEqual([column for instr, line, column in rows], [0, 0, 0])

    def test_py310_minus_match_validation(self):
        """The old text/token matcher only allows line-start matches."""
        old = injector_module._validate_match_at_line_start_py310_minus
        old('    needle = 1', 4, 'needle', 'Location.text', 'text')
        old('        ', 7, 'needle', 'Location.text', 'text')
        with self.assertRaisesRegex(ValueError, 'Python 3.10'):
            old('    x = needle', 8, 'needle', 'Location.text', 'text')

        # The modern helper is deliberately a no-op.
        injector_module._validate_match_at_line_start_py311_plus(
            '    x = needle', 8, 'needle', 'Location.text', 'text')

    def test_py310_minus_text_position_helper(self):
        """The old text helper backs up to the containing token start."""
        class Token:
            start = (3, 7)
        self.assertEqual(
            injector_module._text_position_to_search_start_position_py310_minus(
                100, 20, 15, Token),
            (22, 7))
        self.assertEqual(
            injector_module._text_position_to_search_start_position_py311_plus(
                100, 20, 15, Token),
            (100, 15))

    def test_py310_minus_range_helper_ignores_columns(self):
        """Old source-position matching is line-only, even on current Python."""
        line = sample_function.__code__.co_firstlineno + 1
        old_range = injector_module._find_bytecode_range_for_source_range_py310_minus
        first, stop = old_range(sample_function, line, 999, line, 1000)
        self.assertIsInstance(first, int)
        self.assertGreater(stop, first)
        self.assertEqual(old_range(sample_function, line + 1000, 0, line + 1001, 0),
                         (None, None))

        if _python_3_11_plus:
            new_range = injector_module._find_bytecode_range_for_source_range_py311_plus
            self.assertEqual(
                new_range(sample_function, line, 999, line, 1000),
                (None, None))


    def test_old_insert_call_bytecode_helpers_emit_expected_instruction_shapes(self):
        """Exercise old-bytecode helpers without asking bytecode to validate old opcodes."""
        class FakeInstr:
            def __init__(self, opname, arg=None, *, lineno=None):
                self.opname = opname
                self.arg = arg
                self.lineno = lineno

        old_instr = injector_module.Instr
        try:
            injector_module.Instr = FakeInstr

            bc = []
            injector_module._insert_call_bytecode_py310_minus(bc, 0, 'f', 123)
            self.assertEqual(
                [(i.opname, i.arg, i.lineno) for i in bc],
                [('LOAD_GLOBAL', 'f', 123),
                 ('CALL_FUNCTION', 0, 123),
                 ('POP_TOP', None, 123)])

            bc = []
            injector_module._insert_call_bytecode_py311(bc, 0, 'f', 456)
            self.assertEqual(
                [(i.opname, i.arg, i.lineno) for i in bc],
                [('LOAD_GLOBAL', (True, 'f'), 456),
                 ('PRECALL', 0, 456),
                 ('CALL', 0, 456),
                 ('POP_TOP', None, 456)])
        finally:
            injector_module.Instr = old_instr


    def test_py310_minus_position_helper_runs_on_current_python(self):
        """The old public-position implementation is directly testable."""
        loc = Location._position_py310_minus(sample_function, 2)
        self.assertIsInstance(loc, Location)
        self.assertIs(loc.function, sample_function)

        with self.assertRaisesRegex(ValueError, 'outside function range'):
            Location._position_py310_minus(sample_function, 0)

        with self.assertRaisesRegex(ValueError, 'Python 3.10'):
            Location._position_py310_minus(sample_function, 2, 5)

        source_lines = inspect.getsource(function_with_trailing_uncompiled_line).splitlines()
        trailing_line = len(source_lines)
        with self.assertRaisesRegex(ValueError, 'No bytecode was compiled'):
            Location._position_py310_minus(function_with_trailing_uncompiled_line, trailing_line)


class TestModifyBytecode(unittest.TestCase):

    def test_location_has_start_and_stop(self):
        """Test that Location objects have start and stop attributes."""
        loc = Location.position(sample_function, 1)
        self.assertIsInstance(loc.start, int)
        self.assertIsInstance(loc.stop, int)
        self.assertGreaterEqual(loc.stop, loc.start)

    def test_location_text_wraps_tokenization_errors(self):
        old_getsource = injector_module.inspect.getsource
        try:
            injector_module.inspect.getsource = lambda function: "x = (\n"
            with self.assertRaises(tokenize.TokenError) as cm:
                Location.text(sample_function, "x")
        finally:
            injector_module.inspect.getsource = old_getsource

        message = str(cm.exception)
        self.assertIn("Tokenization failed", message)
        self.assertIn(sample_function.__code__.co_filename, message)

    def test_location_token_wraps_tokenization_errors(self):
        old_getsource = injector_module.inspect.getsource
        try:
            injector_module.inspect.getsource = lambda function: "x = (\n"
            with self.assertRaises(tokenize.TokenError) as cm:
                Location.token(sample_function, "x")
        finally:
            injector_module.inspect.getsource = old_getsource

        message = str(cm.exception)
        self.assertIn("Tokenization failed", message)
        self.assertIn(sample_function.__code__.co_filename, message)

    def test_location_comparison_tuple_like(self):
        """Test Location comparison works like tuple comparison."""
        loc1 = Location(sample_function, 1, 2)
        loc2 = Location(sample_function, 1, 3)
        loc3 = Location(sample_function, 2, 3)

        # Same start, different stop
        self.assertLess(loc1, loc2)
        self.assertGreater(loc2, loc1)

        # Different start
        self.assertLess(loc1, loc3)
        self.assertLess(loc2, loc3)

    def test_location_comparison_different_functions(self):
        """Test that comparing Locations from different functions raises ValueError."""
        loc1 = Location.position(sample_function, 1)
        loc2 = Location.position(do_nothing, 1)

        with self.assertRaises(ValueError) as cm:
            loc1 < loc2
        self.assertIn("different functions", str(cm.exception))

    def test_location_comparison_operators(self):
        """Test that <=, >, >= work correctly for comparing Locations."""
        loc1 = Location(sample_function, 1, 2)
        loc2 = Location(sample_function, 1, 3)
        loc3 = Location(sample_function, 2, 3)

        # Test <=
        self.assertTrue(loc1 <= loc2)
        self.assertTrue(loc1 <= loc1)
        self.assertFalse(loc2 <= loc1)

        # Test >
        self.assertTrue(loc2 > loc1)
        self.assertTrue(loc3 > loc1)
        self.assertFalse(loc1 > loc2)

        # Test >=
        self.assertTrue(loc2 >= loc1)
        self.assertTrue(loc1 >= loc1)
        self.assertTrue(loc3 >= loc1)
        self.assertFalse(loc1 >= loc2)

    def test_location_comparison_not_implemented(self):
        """Test that comparing Location to non-Location returns NotImplemented."""
        loc = Location(sample_function, 1, 2)

        self.assertEqual(loc.__lt__(5), NotImplemented)
        self.assertEqual(loc.__le__(5), NotImplemented)
        self.assertEqual(loc.__gt__(5), NotImplemented)
        self.assertEqual(loc.__ge__(5), NotImplemented)
        self.assertEqual(loc.__eq__(5), NotImplemented)

    def test_location_rich_compare_with_different_functions(self):
        """Test that <=, >, >= with different functions raises ValueError."""
        loc1 = Location.position(sample_function, 1)
        loc2 = Location.position(do_nothing, 1)

        with self.assertRaises(ValueError) as cm:
            loc1 <= loc2
        self.assertIn("different functions", str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            loc1 > loc2
        self.assertIn("different functions", str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            loc1 >= loc2
        self.assertIn("different functions", str(cm.exception))

    def test_location_equality(self):
        """Test Location equality."""
        loc1 = Location(sample_function, 5, 7)
        loc2 = Location(sample_function, 5, 7)
        loc3 = Location(sample_function, 5, 8)

        self.assertEqual(loc1, loc2)
        self.assertNotEqual(loc1, loc3)

    def test_location_hash(self):
        """Test Location can be hashed."""
        loc1 = Location(sample_function, 5, 7)
        loc2 = Location(sample_function, 5, 7)

        # Should be able to use in sets/dicts
        location_set = {loc1, loc2}
        self.assertEqual(len(location_set), 1)

    def test_location_max(self):
        """Test using max() with Location objects."""
        loc1 = Location.position(sample_function, 1)
        loc2 = Location.position(sample_function, 3)
        loc3 = Location.position(sample_function, 2)

        max_loc = max([loc1, loc2, loc3])
        self.assertGreater(max_loc, loc1)
        self.assertGreater(max_loc, loc3)

    def test_location_repr(self):
        """Test Location repr."""
        loc = Location(sample_function, 5, 7)
        repr_str = repr(loc)
        self.assertIn("Location", repr_str)
        self.assertIn("sample_function", repr_str)
        self.assertIn("5", repr_str)
        self.assertIn("7", repr_str)

    def test_location_position_basic(self):
        """Test Location.position with basic line number."""
        loc = Location.position(sample_function, 2)
        self.assertIsInstance(loc, Location)
        self.assertEqual(loc.function, sample_function)
        self.assertGreaterEqual(loc.stop, loc.start)

    if not _python_3_11_plus:
        def test_location_position_column_not_supported_on_old_python(self):
            """Test that Location.position raises on Python 3.10- when column != 1."""
            with self.assertRaises(ValueError) as cm:
                Location.position(sample_function, 2, 5)
            self.assertIn("Python 3.10", str(cm.exception))
            self.assertIn("column", str(cm.exception).lower())

    def test_location_position_out_of_range(self):
        """Test that position outside function range raises ValueError."""
        with self.assertRaises(ValueError):
            Location.position(sample_function, 100)

    def test_strip_line_and_column_information_with_name_parameter(self):
        """Test that _strip_line_and_column_information can change function name."""
        stripped = _strip_line_and_column_information(
            sample_function,
            firstlineno=None,
            name='renamed_function'
        )
        self.assertEqual(stripped.__name__, 'renamed_function')
        self.assertEqual(stripped.__code__.co_name, 'renamed_function')

    def test_strip_line_and_column_information_with_qualname_parameter(self):
        """Test that _strip_line_and_column_information can change qualname."""
        stripped = _strip_line_and_column_information(
            sample_function,
            firstlineno=None,
            qualname='Outer.Inner.renamed_function'
        )
        self.assertEqual(stripped.__qualname__, 'Outer.Inner.renamed_function')
        self.assertEqual(stripped.__code__.co_qualname, 'Outer.Inner.renamed_function')

    def test_location_text_not_found(self):
        """Test that missing text raises ValueError."""
        with self.assertRaises(ValueError) as cm:
            Location.text(sample_function, "nonexistent")
        self.assertIn("not found", str(cm.exception))

    def test_location_text_in_whitespace_does_not_hang(self):
        """Text matches in whitespace should not trap Location.text in its token scan."""
        self.assertIsInstance(Location.text(sample_function, " "), Location)

    def test_location_text_match_after_token_stream_is_exhausted(self):
        """A text match with no token left to contain it is simply not a match."""
        old_getsource = injector_module.inspect.getsource
        old_tokenize = injector_module.tokenize.tokenize
        try:
            injector_module.inspect.getsource = lambda function: "def sample_function():\n    x = 1\n"
            injector_module.tokenize.tokenize = lambda readline: iter(())
            with self.assertRaises(ValueError) as cm:
                Location.text(sample_function, "x")
        finally:
            injector_module.inspect.getsource = old_getsource
            injector_module.tokenize.tokenize = old_tokenize
        self.assertIn("not found", str(cm.exception))

    def test_location_token_tolerates_token_line_outside_source(self):
        """The validation lookup is skipped if a synthetic token reports a bad line."""
        class Token:
            def __init__(self, string, start, type):
                self.string = string
                self.start = start
                self.end = start
                self.type = type

        old_getsource = injector_module.inspect.getsource
        old_tokenize = injector_module.tokenize.tokenize
        old_find_range = injector_module._find_bytecode_range_for_source_range
        old_validate = injector_module._validate_match_at_line_start
        try:
            injector_module.inspect.getsource = lambda function: "def sample_function():\n    pass\n"
            injector_module.tokenize.tokenize = lambda readline: iter((
                Token("needle", (99, 0), tokenize.NAME),
                Token("", (99, 6), tokenize.NEWLINE),
                ))
            injector_module._find_bytecode_range_for_source_range = lambda *args: (0, 1)
            injector_module._validate_match_at_line_start = (
                lambda *args: self.fail("line validation should have been skipped"))
            loc = Location.token(sample_function, "needle")
        finally:
            injector_module.inspect.getsource = old_getsource
            injector_module.tokenize.tokenize = old_tokenize
            injector_module._find_bytecode_range_for_source_range = old_find_range
            injector_module._validate_match_at_line_start = old_validate
        self.assertEqual((loc.start, loc.stop), (0, 1))

    def test_location_token_basic(self):
        """Test Location.token with basic token search."""
        loc = Location.token(sample_function, "x")
        self.assertIsInstance(loc, Location)
        self.assertGreaterEqual(loc.stop, loc.start)

    def test_location_token_not_found(self):
        """Test that missing token raises ValueError."""
        with self.assertRaises(ValueError) as cm:
            Location.token(sample_function, "nonexistent_token")
        self.assertIn("not found", str(cm.exception))

    def test_location_token_with_skip(self):
        """Test Location.token with skip parameter."""
        loc0 = Location.token(sample_function, "return", skip=0)
        loc1 = Location.token(sample_function, "return", skip=1)
        self.assertLess(loc0, loc1)

    def test_location_bytecode_basic(self):
        """Test Location.bytecode with basic bytecode search."""
        loc = Location.bytecode(sample_function, "RETURN_VALUE")
        self.assertIsInstance(loc, Location)
        self.assertEqual(loc.stop, loc.start + 1)

    def test_location_bytecode_not_found(self):
        """Test that missing bytecode raises ValueError."""
        with self.assertRaises(ValueError) as cm:
            Location.bytecode(sample_function, "NONEXISTENT_INSTRUCTION")
        self.assertIn("not found", str(cm.exception))

    def test_location_bytecode_with_skip(self):
        """Test Location.bytecode with skip parameter."""
        loc1 = Location.bytecode(sample_function, "RETURN_VALUE", skip=0)
        loc2 = Location.bytecode(sample_function, "RETURN_VALUE", skip=1)
        self.assertLess(loc1, loc2)

    def test_location_bytecode_skip_out_of_range(self):
        """Test that skip beyond matches raises ValueError for bytecode."""
        with self.assertRaises(ValueError) as cm:
            Location.bytecode(sample_function, "RETURN_VALUE", skip=10)
        self.assertIn("cannot skip", str(cm.exception))

    def test_location_bytecode_with_after(self):
        """Test Location.bytecode with after parameter."""
        # Find first RETURN_VALUE
        first_return = Location.bytecode(sample_function, "RETURN_VALUE", skip=0)
        # Find next RETURN_VALUE after it
        second_return = Location.bytecode(sample_function, "RETURN_VALUE", after=first_return)

        self.assertGreater(second_return, first_return)

    def test_inject_call_basic(self):
        """Test basic call injection."""
        call_count = [0]

        def injected():
            call_count[0] += 1

        loc = Location.position(sample_function, 1)
        modified = inject_call(injected, loc, name="test_func")
        result = modified()

        self.assertEqual(result, 3)
        self.assertEqual(call_count[0], 1)

    def test_inject_call_auto_name(self):
        """Test auto-naming when function name not in globals."""
        call_count = [0]

        def unique_function_name_12345():
            call_count[0] += 1

        self.assertNotIn('unique_function_name_12345', sample_function.__globals__)

        loc = Location.position(sample_function, 1)
        modified = inject_call(unique_function_name_12345, loc)
        result = modified()

        self.assertEqual(result, 3)
        self.assertEqual(call_count[0], 1)

    def test_inject_call_name_collision(self):
        """Test auto-naming with name collisions."""
        call_count = [0]

        def colliding_function():
            call_count[0] += 1

        sample_function.__globals__['colliding_function'] = lambda: None
        sample_function.__globals__['colliding_function_1'] = lambda: None

        try:
            loc = Location.position(sample_function, 1)
            modified = inject_call(colliding_function, loc)
            result = modified()

            self.assertEqual(result, 3)
            self.assertEqual(call_count[0], 1)

            self.assertIn('colliding_function', modified.__globals__)
            self.assertIn('colliding_function_1', modified.__globals__)
            self.assertIn('colliding_function_2', modified.__globals__)
        finally:
            del sample_function.__globals__['colliding_function']
            del sample_function.__globals__['colliding_function_1']

    def test_inject_call_explicit_name_collision(self):
        """Test that explicit name parameter raises ValueError when it already exists."""
        def some_function():
            pass

        some_function()  # Called for coverage

        # Put a name in the globals
        sample_function.__globals__['blammo'] = lambda: None

        try:
            loc = Location.position(sample_function, 1)
            with self.assertRaises(ValueError) as cm:
                inject_call(some_function, loc, name='blammo')
            self.assertIn("already exists", str(cm.exception))
            self.assertIn("blammo", str(cm.exception))
        finally:
            del sample_function.__globals__['blammo']

    def test_inject_call_string_name(self):
        """Test injection using string name from globals."""
        call_count = [0]

        def tracked_function():
            call_count[0] += 1

        sample_function.__globals__['tracked_function'] = tracked_function

        try:
            loc = Location.position(sample_function, 1)
            modified = inject_call('tracked_function', loc)
            result = modified()

            self.assertEqual(result, 3)
            self.assertEqual(call_count[0], 1)
        finally:
            del sample_function.__globals__['tracked_function']

    def test_inject_call_string_name_not_found(self):
        """Test that string name not in globals raises ValueError."""
        loc = Location.position(sample_function, 1)
        with self.assertRaises(ValueError) as cm:
            inject_call('nonexistent_function', loc)
        self.assertIn("not found in function globals", str(cm.exception))

    def test_inject_call_at_end_of_bytecode_uses_no_line_info(self):
        """A Location past the last instruction inserts at the end cleanly."""
        calls = []
        def injected():
            calls.append('called')

        injected()
        self.assertEqual(calls, ['called'])
        calls.clear()

        loc = Location(sample_function, 100000, 100001)
        modified = inject_call(injected, loc, name='end_injection')
        self.assertEqual(modified(), 3)
        self.assertEqual(calls, [])

    def test_sample_function_with_argument(self):
        """Test sample_function with argument to cover conditional branch."""
        result = sample_function(5)
        self.assertEqual(result, 10)

    def test_find_statement_end_with_endmarker(self):
        """Test _find_statement_end when statement ends with ENDMARKER."""
        tokens = [
            tokenize.TokenInfo(tokenize.NAME, 'return', (1, 0), (1, 6), 'return x'),
            tokenize.TokenInfo(tokenize.NAME, 'x', (1, 7), (1, 8), 'return x'),
            tokenize.TokenInfo(tokenize.ENDMARKER, '', (2, 0), (2, 0), ''),
        ]

        result = _find_statement_end(tokens)
        self.assertEqual(result, len(tokens) - 1)

    def test_location_token_endmarker(self):
        """Test that searching for ENDMARKER token raises ValueError."""
        with self.assertRaises(ValueError) as cm:
            Location.token(sample_function, 'ENDMARKER')
        self.assertIn('ENDMARKER', str(cm.exception))
        self.assertIn('bytecode', str(cm.exception))

    def test_inject_call_on_closure(self):
        """A closure (function with free variables) starts with a
        COPY_FREE_VARS / MAKE_CELL prologue instruction whose
        position info is None.  inject_call must skip such prologue
        instructions when locating the injection point and must
        still produce a working modified function."""
        captured = 10

        def closure_fn():
            return captured + 5

        # Sanity check: this IS a closure on Pythons where
        # closures are implemented with cell-based capture.
        self.assertEqual(closure_fn(), 15)
        self.assertGreater(len(closure_fn.__code__.co_freevars), 0)

        call_count = [0]
        def injected():
            call_count[0] += 1

        loc = Location.text(closure_fn, "return captured")
        modified = inject_call(injected, loc)
        result = modified()
        self.assertEqual(result, 15)
        self.assertEqual(call_count[0], 1)

    def test_location_position_on_closure(self):
        """Location.position works on a closure: the prologue
        instruction with no source position is skipped, and the
        first instruction at the requested line is returned."""
        captured = 7

        def closure_fn():
            x = captured
            return x

        loc = Location.position(closure_fn, 2)  # the `x = captured` line
        self.assertIsInstance(loc, Location)
        self.assertGreaterEqual(loc.stop, loc.start)

        call_count = [0]
        def injected():
            call_count[0] += 1

        modified = inject_call(injected, loc)
        self.assertEqual(modified(), 7)
        self.assertEqual(call_count[0], 1)

    def test_corrupt_code_object_check_still_works_on_closure(self):
        """The relaxed validation rejects only functions whose
        instructions ALL lack position info -- not functions that
        merely have a positionless prologue (closures)."""
        # A closure: positionless prologue but real-source instructions
        # afterward.  Must NOT raise.
        captured = 1
        def closure_fn():
            return captured
        self.assertEqual(closure_fn(), 1)
        Location.position(closure_fn, 1)  # should not raise

        # A function whose entire code object has been stripped of
        # position info: still raises.
        stripped = _strip_line_and_column_information(sample_function, firstlineno=None)
        with self.assertRaises(ValueError) as cm:
            Location.position(stripped, 1)
        self.assertIn("corrupt", str(cm.exception))

def run_tests():
    blankettestlib.run(name="blanket.injector", module=__name__)

if __name__ == '__main__':
    run_tests()
    blankettestlib.finish()
