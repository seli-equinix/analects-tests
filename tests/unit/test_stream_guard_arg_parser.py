"""Unit tests for the streaming JSON arg parser.

Verifies the state machine correctly tracks which field a given chunk of
tool-call argument bytes belongs to, across arbitrary chunk boundaries
(the chunk-split-mid-field case is the critical one).
"""
from confucius.core.chat_models.stream_guard.arg_parser import ArgStreamParser


class TestFieldRoutingInOneShot:
    def test_simple_object(self):
        p = ArgStreamParser()
        p.feed('{"command":"')
        assert p.current_field() == "command"
        p.feed('create')
        assert p.current_field() == "command"
        p.feed('","path":"')
        assert p.current_field() == "path"
        p.feed('/workspace/foo.py')
        assert p.current_field() == "path"
        p.feed('","file_text":"hello"}')
        # After final '}', we're back outside.
        assert p.current_field() is None

    def test_field_transition_boundary(self):
        p = ArgStreamParser()
        # Feed everything as one big blob. Field should track through.
        p.feed('{"path":"/workspace/a.py","command":"create"}')
        assert p.current_field() is None


class TestChunkBoundaryPrecision:
    def test_feed_and_split_charges_chars_correctly(self):
        p = ArgStreamParser()
        # Chunk crosses the boundary from path -> command field.
        segments = p.feed_and_split('{"path":"/foo","command":"create"}')
        # Flatten into (field, concatenated_text) groups.
        by_field: dict[str | None, str] = {}
        for fld, frag in segments:
            by_field[fld] = by_field.get(fld, "") + frag
        # The path value chars should all be charged to 'path';
        # the command value chars to 'command'. Structural chars (quotes,
        # commas, braces) go to field=None.
        assert "/foo" in by_field.get("path", "")
        assert "create" in by_field.get("command", "")

    def test_boundary_split_in_middle_of_value(self):
        p = ArgStreamParser()
        p.feed('{"path":"/workspa')
        assert p.current_field() == "path"
        p.feed('ce/foo.py","command":"view"}')
        assert p.current_field() is None


class TestEscapeSequences:
    def test_escaped_quote_in_value(self):
        p = ArgStreamParser()
        p.feed(r'{"path":"/foo\"bar","cmd":"x"}')
        # Parser doesn't crash; field tracking should still resolve to None
        # at the end (outside the object).
        assert p.current_field() is None

    def test_escaped_backslash(self):
        p = ArgStreamParser()
        p.feed(r'{"path":"C:\\Users\\x","cmd":"view"}')
        assert p.current_field() is None

    def test_escape_at_chunk_boundary(self):
        p = ArgStreamParser()
        # '\' at end of chunk, next char in new chunk — must not crash.
        p.feed(r'{"path":"a\\')
        p.feed(r'b","cmd":"x"}')
        assert p.current_field() is None


class TestNestedStructures:
    def test_nested_object_returns_none_for_inner(self):
        p = ArgStreamParser()
        p.feed('{"outer":{"inner":"val"},"top":"ok"}')
        # We don't emit sub-field events for nested objects.
        # By the end we should be back outside.
        assert p.current_field() is None

    def test_nested_array_tolerated(self):
        p = ArgStreamParser()
        p.feed('{"items":[1,2,3],"cmd":"x"}')
        assert p.current_field() is None


class TestPartialFeeds:
    def test_incremental_byte_by_byte(self):
        p = ArgStreamParser()
        text = '{"path":"/workspace/a.py","cmd":"view"}'
        for ch in text:
            p.feed(ch)
        assert p.current_field() is None

    def test_empty_feed_is_safe(self):
        p = ArgStreamParser()
        p.feed("")
        assert p.current_field() is None


class TestMalformedInput:
    def test_no_opening_brace_is_ignored(self):
        p = ArgStreamParser()
        # Garbage prefix + partial object — value string NOT yet closed,
        # so current_field should still be 'path'.
        p.feed('garbage before valid json {"path":"/x')
        assert p.current_field() == "path"

    def test_unbalanced_close_recovers(self):
        p = ArgStreamParser()
        p.feed('{"path":"/x"}')
        # Second close brace — should not crash.
        p.feed('}')
        assert p.current_field() is None
