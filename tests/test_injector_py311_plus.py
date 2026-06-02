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

import inspect
import sys
import unittest

import blankettestlib
blankettestlib.preload_local_blanket()

from blanket import Location, inject_call
from test_injector import sample_function, _strip_line_and_column_information


global_value = 0


def add_100_to_global_value():
    global global_value
    global_value += 100


if sys.version_info >= (3, 11):

    class TestInjectorPy311Plus(unittest.TestCase):
        """Tests that need Python 3.11+ source-column information."""

        def test_location_position_with_column(self):
            """Test Location.position with line and column."""
            loc = Location.position(sample_function, 2, 5)
            self.assertIsInstance(loc, Location)
            self.assertGreaterEqual(loc.stop, loc.start)


        def test_location_position_past_all_code_by_column(self):
            """Test Location.position when column is past all code on last line."""
            def simple_function():
                x = 1
                return x

            simple_function()  # Called for coverage

            # Get the number of lines in the function
            source_lines = inspect.getsource(simple_function).splitlines()
            last_line = len(source_lines)

            # Request a position on the last line at column 999
            with self.assertRaises(ValueError) as cm:
                Location.position(simple_function, last_line, 999)
            self.assertIn("No bytecode", str(cm.exception))


        def test_location_position_with_corrupt_code_object(self):
            """Test that Location.position raises on functions with corrupt code objects."""
            # Test with invalid firstlineno
            stripped_invalid = _strip_line_and_column_information(sample_function, firstlineno=-1)
            with self.assertRaises(ValueError) as cm:
                Location.position(stripped_invalid, 1)
            self.assertIn("corrupt", str(cm.exception))

            # Test with missing line information
            stripped_no_lines = _strip_line_and_column_information(sample_function, firstlineno=None)
            with self.assertRaises(ValueError) as cm:
                Location.position(stripped_no_lines, 1)
            self.assertIn("corrupt", str(cm.exception))


        def test_location_text_basic(self):
            """Test Location.text with simple text search."""
            loc = Location.text(sample_function, "x = 1")
            self.assertIsInstance(loc, Location)
            self.assertEqual(loc.function, sample_function)
            self.assertGreaterEqual(loc.stop, loc.start)


        def test_location_text_in_comment_not_found(self):
            """Test that text in comment raises ValueError (no bytecode for comments)."""
            def function_with_comment():
                x = 1  # unique comment marker text
                return x

            self.assertEqual(function_with_comment(), 1)

            with self.assertRaises(ValueError) as cm:
                Location.text(function_with_comment, "unique comment marker")
            self.assertIn("not found", str(cm.exception))


        def test_location_text_in_docstring_not_found(self):
            """Test that text in docstring raises ValueError (no bytecode for docstrings)."""
            def function_with_docstring():
                """This docstring contains rubber baby buggy bumpers."""
                x = 1
                return x

            self.assertEqual(function_with_docstring(), 1)

            with self.assertRaises(ValueError) as cm:
                Location.text(function_with_docstring, "rubber baby buggy bumpers")
            self.assertIn("not found", str(cm.exception))


        def test_location_text_with_skip(self):
            """Test Location.text with skip parameter."""
            def function_with_duplicates():
                global global_value
                x = 1 - global_value
                y = 1 - global_value
                return x + y

            # Verify unmodified behavior
            global global_value
            global_value = 0
            self.assertEqual(function_with_duplicates(), 2)

            # Inject before first "= 1"
            global_value = 0
            loc0 = Location.text(function_with_duplicates, "= 1", skip=0)
            modified0 = inject_call(add_100_to_global_value, loc0)
            self.assertEqual(modified0(), -198)
            self.assertEqual(global_value, 100)

            # Inject before second "= 1"
            global_value = 0
            loc1 = Location.text(function_with_duplicates, "= 1", skip=1)
            modified1 = inject_call(add_100_to_global_value, loc1)
            self.assertEqual(modified1(), -98)
            self.assertEqual(global_value, 100)


        def test_location_text_skip_out_of_range(self):
            """Test that skip beyond matches raises ValueError."""
            with self.assertRaises(ValueError) as cm:
                Location.text(sample_function, "x = 1", skip=5)
            self.assertIn("cannot skip", str(cm.exception))


        def test_location_text_with_after(self):
            """Test Location.text with after parameter."""
            global global_value
            global_value = 0

            def function_with_duplicates():
                global global_value
                x = 1 - global_value
                z = 1
                y = 2 - global_value
                z -= global_value
                return (x, y, z)

            self.assertEqual(function_with_duplicates(), (1, 2, 1))

            # Find location of "y", then find "z" after that
            y = Location.text(function_with_duplicates, "y")
            loc = Location.text(function_with_duplicates, "z", after=y)

            # Inject before the "z -=" line
            modified = inject_call(add_100_to_global_value, loc)
            result = modified()
            self.assertEqual(result, (1, 2, -99))


        def test_location_text_with_after_using_max(self):
            """Test Location.text with after using max() of multiple constraints."""
            global global_value
            global_value = 0

            def function_with_duplicates():
                global global_value
                x = 1 - global_value
                z = 1
                y = 2 - global_value
                z -= global_value
                return (x, y, z)

            # Find "z" after the maximum of text "y" and text "2"
            y_and_2 = max([
                Location.text(function_with_duplicates, "y"),
                Location.text(function_with_duplicates, "2")
            ])
            loc = Location.text(function_with_duplicates, "z", after=y_and_2)

            modified = inject_call(add_100_to_global_value, loc)
            result = modified()
            self.assertEqual(result, (1, 2, -99))


        def test_location_token_skip_out_of_range(self):
            """Test that skip beyond matches raises ValueError for token."""
            with self.assertRaises(ValueError) as cm:
                Location.token(sample_function, "return", skip=10)
            self.assertIn("cannot skip", str(cm.exception))


        def test_location_token_with_after_regression(self):
            """Regression test: token after text should respect text position within line."""
            global global_value
            global_value = 0

            def crazy_function(a):
                global global_value
                if a > 3:
                    return (a * 2) - global_value
                foo = 1
                return (a * 3) - global_value

            # Verify unmodified behavior
            self.assertEqual(crazy_function(5), 10)
            self.assertEqual(crazy_function(2), 6)

            # Search for 'return' token after text "foo"
            foo_location = Location.text(crazy_function, "foo")
            loc = Location.token(crazy_function, 'return', after=foo_location)

            # Inject before the second return (the one in the else path)
            modified = inject_call(add_100_to_global_value, loc)

            # First branch shouldn't trigger injection
            global_value = 0
            self.assertEqual(modified(5), 10)
            self.assertEqual(global_value, 0)

            # Second branch should trigger injection
            global_value = 0
            self.assertEqual(modified(2), -94)
            self.assertEqual(global_value, 100)


        def test_location_token_multiline(self):
            """Test Location.token with a multi-line token."""
            global global_value
            global_value = 0

            def function_with_multiline_strings():
                global global_value

                jabberwocky = '''
    'Twas brillig, and the slithy toves
    Did gyre and gimble in the wabe:
    All mimsy were the borogoves,
    And the mome raths outgrabe.
    '''.ljust(3)
                jabberwocky_number = len(jabberwocky) - global_value

                the_crocodile = '''
    How doth the little crocodile
         Improve his shining tail,
    And pour the waters of the Nile
         On every golden scale!
    '''.rjust(
        3)
                the_crocodile_number = len(the_crocodile) - global_value

                return (jabberwocky, jabberwocky_number, the_crocodile, the_crocodile_number)

            # Call unmodified function to get the strings
            jabberwocky, jabberwocky_number, the_crocodile, the_crocodile_number = function_with_multiline_strings()

            # Verify unmodified behavior.  The exact lengths depend on
            # indentation, so compute them rather than hard-coding them.
            self.assertEqual((jabberwocky_number, the_crocodile_number),
                (len(jabberwocky), len(the_crocodile)))

            # Find the first multi-line string token and inject before it
            loc = Location.token(function_with_multiline_strings, f"'''{jabberwocky}'''")
            modified = inject_call(add_100_to_global_value, loc)

            global_value = 0
            result = modified()
            self.assertEqual((result[1], result[3]),
                (jabberwocky_number - 100, the_crocodile_number - 100))
            self.assertEqual(global_value, 100)

            # Find "the" after the first multiline string
            the_loc = Location.text(function_with_multiline_strings, 'the', after=loc)
            modified2 = inject_call(add_100_to_global_value, the_loc)

            global_value = 0
            result2 = modified2()
            self.assertEqual((result2[1], result2[3]),
                (jabberwocky_number, the_crocodile_number - 100))
            self.assertEqual(global_value, 100)

            # Search for "ljust" using Location.text
            ljust_text_loc = Location.text(function_with_multiline_strings, 'ljust')
            modified3 = inject_call(add_100_to_global_value, ljust_text_loc)

            global_value = 0
            result3 = modified3()
            self.assertEqual((result3[1], result3[3]), (jabberwocky_number - 100, the_crocodile_number - 100))
            self.assertEqual(global_value, 100)

            # Search for "ljust" using Location.token
            ljust_token_loc = Location.token(function_with_multiline_strings, 'ljust')
            modified4 = inject_call(add_100_to_global_value, ljust_token_loc)

            global_value = 0
            result4 = modified4()
            self.assertEqual((result4[1], result4[3]), (jabberwocky_number - 100, the_crocodile_number - 100))
            self.assertEqual(global_value, 100)

            # Search for "rjust" using Location.text to test multi-line range
            rjust_text_loc = Location.text(function_with_multiline_strings, 'rjust')
            modified5 = inject_call(add_100_to_global_value, rjust_text_loc)

            global_value = 0
            result5 = modified5()
            self.assertEqual((result5[1], result5[3]), (jabberwocky_number, the_crocodile_number - 100))
            self.assertEqual(global_value, 100)

            # Search for "rjust" using Location.token
            rjust_token_loc = Location.token(function_with_multiline_strings, 'rjust')
            modified6 = inject_call(add_100_to_global_value, rjust_token_loc)

            global_value = 0
            result6 = modified6()
            self.assertEqual((result6[1], result6[3]), (jabberwocky_number, the_crocodile_number - 100))
            self.assertEqual(global_value, 100)


        def test_inject_call_at_text_location(self):
            """Test injection at a text-based location."""
            call_count = [0]

            def injected():
                call_count[0] += 1

            loc = Location.text(sample_function, "y = 2")
            modified = inject_call(injected, loc)
            result = modified()

            self.assertEqual(result, 3)
            self.assertEqual(call_count[0], 1)


        def test_location_text_on_closure(self):
            """Location.text works on a closure even though its first
            bytecode instruction (COPY_FREE_VARS) has no position info."""
            captured = 100

            def closure_fn():
                y = captured
                return y - 50

            self.assertEqual(closure_fn(), 50)

            # Two distinct matches; verify both work despite the prologue.
            loc1 = Location.text(closure_fn, "captured")
            loc2 = Location.text(closure_fn, "- 50")
            self.assertLess(loc1, loc2)




def run_tests():
    blankettestlib.run(name="blanket.injector.py311_plus", module=__name__)


if __name__ == '__main__':
    run_tests()
    blankettestlib.finish()
