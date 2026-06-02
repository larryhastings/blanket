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


from bytecode import Bytecode, Instr
from collections import deque
from types import FunctionType
import dis
import inspect
import io
import sys
import tokenize

__all__ = []
def export(o):
    if isinstance(o, str):
        s = o
    else:
        s = o.__name__
    __all__.append(s)
    return o


# Version detection for feature availability.
_python_3_11_plus = sys.version_info >= (3, 11)
_python_3_12_plus = sys.version_info >= (3, 12)


# Version-specific implementations.
#
# Define both sides of compatibility branches where the implementation
# can run on modern Python, then bind the public helper name once at
# module import time.  This keeps the hot path free of per-call version
# checks, preserves the public helper's __name__ for introspection, and
# lets tests call the old implementation directly when it is not tied to
# old-interpreter bytecode opcodes.


def _instr_line_column(function):
    """
    Yield (instruction, line, column) tuples for Python 3.10-.

    Provides accurate line information but column is always 0.
    Line numbers are carried forward when an instruction doesn't have
    starts_line.
    """
    line = function.__code__.co_firstlineno

    for instr in dis.get_instructions(function):
        starts_line = instr.starts_line
        if isinstance(starts_line, bool):
            # Python 3.13 changed starts_line from a line-number-or-None
            # value to a boolean, with the actual line in line_number.
            # Preserve the Python 3.10-style carry-forward semantics when
            # this old helper is tested on modern Python.
            if starts_line:
                line = instr.line_number
        elif starts_line is not None:
            line = starts_line
        yield (instr, line, 0)

_instr_line_column_py310_minus = _instr_line_column


def _instr_line_column(function):
    """Yield (instruction, line, column) tuples for Python 3.11+.

    Closures' prologue instructions (COPY_FREE_VARS, MAKE_CELL)
    have no source position; emit None for line/column on those so
    callers can skip them.  The code object is considered corrupt only
    if NO instruction has position info.
    """
    line = function.__code__.co_firstlineno

    if line < 1:
        raise ValueError(f"Function has corrupt code object: co_firstlineno={line}")

    # Validate the code object has at least one instruction with
    # position info.  A closure's prologue (COPY_FREE_VARS, etc.) has
    # no source position, which is normal; but a fabricated function
    # with no positions anywhere is what this check guards against.
    for instr in dis.get_instructions(function):
        positions = instr.positions
        if positions.lineno is not None and positions.col_offset is not None:
            break
    else:
        raise ValueError(
            f"Function has corrupt code object: no instruction has position information. "
            f"If you created this function using a library like bytecode, use Location.bytecode() "
            f"instead of Location.text(), Location.token(), or Location.position()."
        )

    # Yield every instruction (including prologue ones with None
    # positions) so caller indices align with Bytecode.from_code list
    # order.  Callers skip None-positioned entries when checking
    # against source ranges.
    for instr in dis.get_instructions(function):
        positions = instr.positions
        yield (instr, positions.lineno, positions.col_offset)

_instr_line_column_py311_plus = _instr_line_column
_instr_line_column = (_instr_line_column_py311_plus
                      if _python_3_11_plus else
                      _instr_line_column_py310_minus)


def _validate_match_at_line_start(source_line, column, search_string, caller_name, search_type):
    """Validate that match is at line start for Python 3.10-."""
    stripped = source_line.lstrip()
    if not stripped:
        # Empty line or whitespace-only.
        return
    first_non_ws_col = len(source_line) - len(stripped)

    if column != first_non_ws_col:
        raise ValueError(
            f"On Python 3.10 and earlier, {caller_name}() "
            f"can only search for {search_type} at the beginning of a line (after indentation). "
            f"Searching for '{search_string}' at column {column} is not supported. "
            f"Use Python 3.11+ for fine-grained column matching."
        )

_validate_match_at_line_start_py310_minus = _validate_match_at_line_start


def _validate_match_at_line_start(source_line, column, search_string, caller_name, search_type):
    """Validate match position for Python 3.11+ (no restrictions)."""

_validate_match_at_line_start_py311_plus = _validate_match_at_line_start
_validate_match_at_line_start = (_validate_match_at_line_start_py311_plus
                                 if _python_3_11_plus else
                                 _validate_match_at_line_start_py310_minus)


def _text_position_to_search_start_position(absolute_line, function_start_line, match_start_col, containing_token):
    """Convert text match position to bytecode search start position for Python 3.10-."""
    search_start_line = function_start_line + containing_token.start[0] - 1
    search_start_col = containing_token.start[1]
    return (search_start_line, search_start_col)

_text_position_to_search_start_position_py310_minus = _text_position_to_search_start_position


def _text_position_to_search_start_position(absolute_line, function_start_line, match_start_col, containing_token):
    """Convert text match position to bytecode search start position for Python 3.11+."""
    return (absolute_line, match_start_col)

_text_position_to_search_start_position_py311_plus = _text_position_to_search_start_position
_text_position_to_search_start_position = (_text_position_to_search_start_position_py311_plus
                                           if _python_3_11_plus else
                                           _text_position_to_search_start_position_py310_minus)


def _find_bytecode_range_for_source_range(function, start_line, start_col, end_line, end_col):
    """
    Find bytecode instructions whose source positions fall within the
    given source range, and return the span from the earliest to the
    latest such instruction.

    Python 3.10- version with line-level only checking.  Columns are
    intentionally ignored.
    """
    bc = _instr_line_column_py310_minus(function)
    matching_offsets = []

    for i, (instr, instr_line, instr_col) in enumerate(bc):
        if start_line <= instr_line <= end_line:
            matching_offsets.append(i)

    if not matching_offsets:
        return (None, None)

    return (min(matching_offsets), max(matching_offsets) + 1)

_find_bytecode_range_for_source_range_py310_minus = _find_bytecode_range_for_source_range


def _find_bytecode_range_for_source_range(function, start_line, start_col, end_line, end_col):
    """
    Find bytecode instructions whose source positions fall within the
    given source range, and return the span from the earliest to the
    latest such instruction.

    Python 3.11+ version with precise column checking.
    """
    bc = _instr_line_column_py311_plus(function)
    matching_offsets = []

    for i, (instr, instr_line, instr_col) in enumerate(bc):
        # Prologue instructions (COPY_FREE_VARS, etc.) carry no source
        # position; they can't be part of any source range.
        if instr_line is None or instr_col is None:
            continue
        # Position (line, col) is in range if:
        # start <= position < end (lexicographically)
        if start_line < instr_line < end_line:
            in_range = True
        elif instr_line == start_line and instr_line == end_line:
            in_range = start_col <= instr_col < end_col
        elif instr_line == start_line:
            in_range = instr_col >= start_col
        elif instr_line == end_line:
            in_range = instr_col < end_col
        else:
            continue

        if in_range:
            matching_offsets.append(i)

    if not matching_offsets:
        return (None, None)

    return (min(matching_offsets), max(matching_offsets) + 1)

_find_bytecode_range_for_source_range_py311_plus = _find_bytecode_range_for_source_range
_find_bytecode_range_for_source_range = (_find_bytecode_range_for_source_range_py311_plus
                                         if _python_3_11_plus else
                                         _find_bytecode_range_for_source_range_py310_minus)


# The injected-call bytecode is genuinely interpreter-specific: old
# opcodes are not valid for the current bytecode package on new Python.
# Keep these helpers in the same binding-time shape.  Unit tests can
# verify their instruction shapes with fake Instr objects, but full
# integration coverage still belongs on the interpreters whose bytecode
# actually supports those opcodes.
def _insert_call_bytecode(bc, offset, func_name, line_info):
    """Insert function call bytecode for Python 3.10 and earlier."""
    bc.insert(offset, Instr("LOAD_GLOBAL", func_name, lineno=line_info))
    bc.insert(offset + 1, Instr("CALL_FUNCTION", 0, lineno=line_info))
    bc.insert(offset + 2, Instr("POP_TOP", lineno=line_info))

_insert_call_bytecode_py310_minus = _insert_call_bytecode


def _insert_call_bytecode(bc, offset, func_name, line_info):
    """Insert function call bytecode for Python 3.11 only."""
    bc.insert(offset, Instr("LOAD_GLOBAL", (True, func_name), lineno=line_info))
    bc.insert(offset + 1, Instr("PRECALL", 0, lineno=line_info))
    bc.insert(offset + 2, Instr("CALL", 0, lineno=line_info))
    bc.insert(offset + 3, Instr("POP_TOP", lineno=line_info))

_insert_call_bytecode_py311 = _insert_call_bytecode


def _insert_call_bytecode(bc, offset, func_name, line_info):
    """Insert function call bytecode for Python 3.12+ (PRECALL removed)."""
    bc.insert(offset, Instr("LOAD_GLOBAL", (True, func_name), lineno=line_info))
    bc.insert(offset + 1, Instr("CALL", 0, lineno=line_info))
    bc.insert(offset + 2, Instr("POP_TOP", lineno=line_info))

_insert_call_bytecode_py312_plus = _insert_call_bytecode

_insert_call_bytecode_options = (_insert_call_bytecode_py310_minus, _insert_call_bytecode_py311, _insert_call_bytecode_py312_plus)
_insert_call_bytecode = _insert_call_bytecode_options[int(_python_3_11_plus) + int(_python_3_12_plus)]
del _insert_call_bytecode_options


def _find_statement_end(tokens):
    """
    Find the end of the statement in the token sequence.
    
    Returns the index of the NEWLINE or ENDMARKER token that ends the statement.
    """
    for i, tok in enumerate(tokens):
        if tok.type == tokenize.NEWLINE:
            return i
    last_token_index = len(tokens) - 1
    assert tokens[last_token_index].type == tokenize.ENDMARKER
    return last_token_index


@export
class Location:
    """
    Represents a range of bytecode instructions in a function where code can be injected.
    
    Attributes:
        function: The function to inject into
        start: The starting bytecode offset (inclusive)
        stop: The ending bytecode offset (exclusive, like range())
    """
    
    def __init__(self, function, start, stop):
        self.function = function
        self.start = start
        self.stop = stop
    
    def _rich_comparison_precheck(self, other):
        """Validate that other is a Location from the same function."""
        if not isinstance(other, Location):
            return False
        if self.function is not other.function:
            raise ValueError("Cannot compare Locations from different functions")
        return True
    
    def __lt__(self, other):
        if not self._rich_comparison_precheck(other):
            return NotImplemented
        return (self.start, self.stop) < (other.start, other.stop)
    
    def __le__(self, other):
        if not self._rich_comparison_precheck(other):
            return NotImplemented
        return (self.start, self.stop) <= (other.start, other.stop)
    
    def __gt__(self, other):
        if not self._rich_comparison_precheck(other):
            return NotImplemented
        return (self.start, self.stop) > (other.start, other.stop)
    
    def __ge__(self, other):
        if not self._rich_comparison_precheck(other):
            return NotImplemented
        return (self.start, self.stop) >= (other.start, other.stop)
    
    def __eq__(self, other):
        if not isinstance(other, Location):
            return NotImplemented
        return (self.function is other.function) and (self.start == other.start) and (self.stop == other.stop)
    
    def __hash__(self):
        return hash((id(self.function), self.start, self.stop))
    
    def __repr__(self):
        return f"Location({self.function.__name__}, {self.start}, {self.stop})"
    
    @classmethod
    def position(cls, function, line, column=1):
        """
        Find injection location by line position on Python 3.10 and earlier.

        Python 3.10 and earlier expose line information but not useful
        source columns, so only column 1 is supported.
        """
        source_lines = inspect.getsource(function).splitlines()
        if not (1 <= line <= len(source_lines)):
            raise ValueError(f"Line {line} is outside function range (1-{len(source_lines)})")

        if column != 1:
            raise ValueError(
                f"On Python 3.10 and earlier, Location.position() only supports column=1. "
                f"Column {column} is not supported. Use Python 3.11+ for fine-grained column positioning."
            )

        absolute_line = function.__code__.co_firstlineno + line - 1
        for i, (instr, instr_line, instr_col) in enumerate(_instr_line_column_py310_minus(function)):
            if instr_line >= absolute_line:
                return cls(function, i, i + 1)

        raise ValueError(f"No bytecode was compiled from text at line {line}, column {column}")

    _position_py310_minus = position

    @classmethod
    def position(cls, function, line, column=1):
        """
        Find injection location by line and column position.

        Args:
            function: The function to inject into
            line: Relative line number within the function (1-based)
            column: Column within the line (1-based, default 1)

        Returns:
            Location object representing the single instruction at that position
        """
        source_lines = inspect.getsource(function).splitlines()
        if not (1 <= line <= len(source_lines)):
            raise ValueError(f"Line {line} is outside function range (1-{len(source_lines)})")

        absolute_line = function.__code__.co_firstlineno + line - 1
        for i, (instr, instr_line, instr_col) in enumerate(_instr_line_column_py311_plus(function)):
            # Prologue instructions (closure setup) carry no source
            # position; they can't satisfy a line/column query.
            if instr_line is None or instr_col is None:
                continue
            if (instr_line == absolute_line and instr_col >= column) or (instr_line > absolute_line):
                return cls(function, i, i + 1)

        raise ValueError(f"No bytecode was compiled from text at line {line}, column {column}")

    _position_py311_plus = position
    position = _position_py311_plus if _python_3_11_plus else _position_py310_minus

    @classmethod
    def text(cls, function, text, *, skip=0, after=None):
        """
        Find injection location by searching for text in source.
        
        The returned Location spans from the first bytecode instruction that implements
        any source text at or after the matched text in the same statement, up to and
        including the bytecode that directly implements the matched text.
        
        On Python 3.10 and earlier, text can only be matched at the beginning of lines
        (after indentation). Multiple matches on the same line cannot be distinguished.
        
        Args:
            function: The function to inject into
            text: Text to search for
            skip: Number of matching occurrences to skip (default 0)
            after: Location object - search only after this location (default None)
        
        Returns:
            Location object representing the bytecode range
        """
        # Calculate minimum offset
        min_offset = after.stop if after else 0
        
        source = inspect.getsource(function)
        function_start_line = function.__code__.co_firstlineno
        
        # Tokenize the source
        try:
            tokens = deque(tokenize.tokenize(io.BytesIO(source.encode('utf-8')).readline))
        except tokenize.TokenError as e:
            raise tokenize.TokenError(
                f"Tokenization failed for {function.__code__.co_filename}. "
                f"Did you change the file while this test was running? "
                f"Original error: {e}"
            ) from None
        
        # Find all occurrences of the text in the source
        source_lines = source.splitlines()
        original_skip = skip
        match = None
        
        for line_idx, line in enumerate(source_lines):
            col = line.find(text)
            while col != -1:
                absolute_line = function_start_line + line_idx
                match_start_col = col
                
                # Validate match position for Python 3.10-
                _validate_match_at_line_start(line, match_start_col, text, "Location.text", "text")
                
                # Find which token contains this text (consuming from deque)
                containing_token = None
                while tokens:
                    tok = tokens[0]  # Peek
                    tok_line = function_start_line + tok.start[0] - 1
                    tok_start_col = tok.start[1]
                    tok_end_col = tok.end[1]
                    
                    if tok_line < absolute_line or (tok_line == absolute_line and tok_end_col <= match_start_col):
                        # Token is before our match, discard it
                        tokens.popleft()
                    elif tok_line > absolute_line or tok_start_col > match_start_col:
                        # Token is after our match; the match is not inside a token.
                        break
                    else:
                        # The earlier branches rejected tokens before or after
                        # the match; whatever remains must contain it.
                        assert tok_line == absolute_line
                        assert tok_start_col <= match_start_col < tok_end_col
                        containing_token = tokens.popleft()
                        break
                
                if containing_token:
                    # Find end of statement in remaining tokens
                    stmt_end_idx = _find_statement_end(tokens)
                    stmt_end_token = tokens[stmt_end_idx]
                    stmt_end_line = function_start_line + stmt_end_token.start[0] - 1
                    stmt_end_col = stmt_end_token.start[1]
                    
                    # Determine search start position
                    search_start_line, search_start_col = _text_position_to_search_start_position(
                        absolute_line, function_start_line, match_start_col, containing_token
                    )
                    
                    # Find all bytecode from search start to end of statement
                    first, last = _find_bytecode_range_for_source_range(
                        function, search_start_line, search_start_col, stmt_end_line, stmt_end_col
                    )
                    
                    if first is not None and first >= min_offset:
                        if not skip:
                            match = (first, last)
                            break  # Break out of while loop
                        skip -= 1
                
                col = line.find(text, col + 1)
            
            if match:
                break
        else:
            # Didn't find or didn't find enough
            if skip == original_skip:
                raise ValueError(f"text '{text}' not found in function source")
            
            matches_found = original_skip - skip
            raise ValueError(f"text '{text}' found {matches_found} times, cannot skip {original_skip}")
        
        start, stop = match
        return cls(function, start, stop)
    
    @classmethod
    def token(cls, function, token, *, skip=0, after=None):
        """
        Find injection location by searching for a token.
        
        The returned Location spans from the first bytecode instruction that implements
        any source text at or after the matched token in the same statement, up to and
        including the bytecode that directly implements the matched token.
        
        On Python 3.10 and earlier, tokens can only be matched at the beginning of lines
        (after indentation). Multiple matches on the same line cannot be distinguished.
        
        Args:
            function: The function to inject into
            token: Token to search for
            skip: Number of matching occurrences to skip (default 0)
            after: Location object - search only after this location (default None)
        
        Returns:
            Location object representing the bytecode range
        """
        # Reject ENDMARKER - it never produces bytecode
        if token == 'ENDMARKER':
            raise ValueError("can't search for ENDMARKER, this token never produces bytecode")
        
        # Calculate minimum offset
        min_offset = after.stop if after else 0
        
        source = inspect.getsource(function)
        source_lines = source.splitlines()
        function_start_line = function.__code__.co_firstlineno
        
        try:
            tokens = deque(tokenize.tokenize(io.BytesIO(source.encode('utf-8')).readline))
        except tokenize.TokenError as e:
            raise tokenize.TokenError(
                f"Tokenization failed for {function.__code__.co_filename}. "
                f"Did you change the file while this test was running? "
                f"Original error: {e}"
            ) from None
        
        original_skip = skip
        match = None
        
        while tokens:
            tok = tokens.popleft()
            
            if tok.string != token:
                continue
            
            # Found a matching token
            token_start_line = function_start_line + tok.start[0] - 1
            token_start_col = tok.start[1]
            
            # Validate match position for Python 3.10-
            source_line_idx = tok.start[0] - 1
            if 0 <= source_line_idx < len(source_lines):
                _validate_match_at_line_start(source_lines[source_line_idx], token_start_col, token, "Location.token", "tokens")
            
            # Find end of statement - tok has been popped, so search remaining tokens from index 0
            stmt_end_idx = _find_statement_end(tokens)
            stmt_end_token = tokens[stmt_end_idx]
            stmt_end_line = function_start_line + stmt_end_token.start[0] - 1
            stmt_end_col = stmt_end_token.start[1]
            
            # Find all bytecode from token position to end of statement
            first, last = _find_bytecode_range_for_source_range(
                function, token_start_line, token_start_col, stmt_end_line, stmt_end_col
            )
            
            if first is not None and first >= min_offset:
                if not skip:
                    match = (first, last)
                    break
                skip -= 1
        else:
            # Didn't find or didn't find enough
            if skip == original_skip:
                raise ValueError(f"token '{token}' not found in function source")
            
            matches_found = original_skip - skip
            raise ValueError(f"token '{token}' found {matches_found} times, cannot skip {original_skip}")
        
        start, stop = match
        return cls(function, start, stop)
    
    @classmethod
    def bytecode(cls, function, bytecode, *, skip=0, after=None):
        """
        Find injection location by searching for a bytecode instruction.
        
        Args:
            function: The function to inject into
            bytecode: Bytecode instruction name to search for
            skip: Number of matching occurrences to skip (default 0)
            after: Location object - search only after this location (default None)
        
        Returns:
            Location object representing that single bytecode instruction
        """
        # Calculate minimum offset
        min_offset = after.stop if after else 0
        
        # Find matching bytecode instructions at or after min_offset
        bc = _instr_line_column(function)
        original_skip = skip
        
        for i, (instr, line, column) in enumerate(bc):
            if instr.opname == bytecode and i >= min_offset:
                if not skip:
                    break
                skip -= 1
        else:
            # Didn't find or didn't find enough
            if skip == original_skip:
                raise ValueError(f"bytecode '{bytecode}' not found in function")
            
            matches_found = original_skip - skip
            raise ValueError(f"bytecode '{bytecode}' found {matches_found} times, cannot skip {original_skip}")
        
        start = i
        stop = start + 1
        return cls(function, start, stop)


@export
def inject_call(injected_function, location, *, name=''):
    """
    Create a new function that calls injected_function at the specified location.
    
    Parameters:
        injected_function: Either a callable to inject, or a string name to look up in globals
        location: Location object specifying where to inject
        name: Name to bind injected_function to in new function's globals (keyword-only)
              If empty string (default), auto-generate a name
    
    Returns:
        A new function object with the injected call.
    """
    function = location.function
    offset = location.start
    
    # Determine the name and globals to use
    if callable(injected_function):
        # If user provided a name, use it
        if name:
            new_globals = function.__globals__.copy()
            if name in new_globals:
                raise ValueError(f"Name '{name}' already exists in function globals")
            new_globals[name] = injected_function
        # Check if already bound correctly in original globals
        elif function.__globals__.get(injected_function.__name__) is injected_function:
            # Already bound correctly, use original globals as-is
            new_globals = function.__globals__
            name = injected_function.__name__
        else:
            # Need to bind it - make a copy of globals
            new_globals = function.__globals__.copy()
            
            # Auto-generate name
            name = injected_function.__name__
            n = 0
            while name in new_globals:
                n += 1
                name = f"{injected_function.__name__}_{n}"
            
            new_globals[name] = injected_function
    else:
        # String name - look it up in globals
        new_globals = function.__globals__
        name = injected_function
        if name not in function.__globals__:
            raise ValueError(f"Name '{name}' not found in function globals")
    
    # Get the bytecode using the bytecode library
    bc = Bytecode.from_code(function.__code__)

    # Location offsets count instructions only (from dis.get_instructions),
    # but bc indexes include Label items.  Convert the dis-index 'offset'
    # to the matching bc-list index by counting Instr items until we
    # reach the right one.
    instr_count = 0
    bc_offset = len(bc)
    for bc_idx, item in enumerate(bc):
        if isinstance(item, Instr):
            if instr_count == offset:
                bc_offset = bc_idx
                break
            instr_count += 1

    # Get line info for the injection point
    line_info = None
    if bc_offset < len(bc) and isinstance(bc[bc_offset], Instr):
        line_info = bc[bc_offset].lineno

    # Insert function call bytecode
    _insert_call_bytecode(bc, bc_offset, name, line_info)
    
    # Create new code object
    new_code = bc.to_code()
    
    # Create new function with modified bytecode and new globals
    new_function = FunctionType(
        new_code,
        new_globals,
        function.__name__,
        function.__defaults__,
        function.__closure__
    )
    
    return new_function

