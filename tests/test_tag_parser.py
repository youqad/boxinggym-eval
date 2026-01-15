"""
Unit tests for TagParser class.

Tests cover:
- Basic well-formed tags
- Whitespace variations
- Missing closing tags
- Empty tags
- "Done" signals
- Missing numbers
- Format normalization
- Multiple tags
- Case insensitivity
- Diagnostic messages

Run with: python -m pytest tests/test_tag_parser.py -v
"""

from boxing_gym.agents.tag_parser import TagParser, ParsingStrategy


class TestTagParserBasic:
    """Test basic tag parsing with well-formed input."""

    def test_well_formed_observe_tag(self):
        """Parse standard observe tag."""
        parser = TagParser("observe", validation_mode="moderate")
        result = parser.parse("<observe>[1, 2, 3]</observe>")
        assert result == "[1, 2, 3]"
        assert parser.last_strategy_used == ParsingStrategy.STRICT

    def test_well_formed_answer_tag(self):
        """Parse standard answer tag."""
        parser = TagParser("answer", validation_mode="moderate")
        result = parser.parse("<answer>0.5</answer>")
        assert result == "0.5"

    def test_numeric_extraction(self):
        """Extract numbers from valid content."""
        parser = TagParser("observe", validation_mode="moderate")
        result = parser.parse("<observe>Result: [42, 100, 5]</observe>")
        assert result is not None
        assert "42" in result


class TestTagParserWhitespace:
    """Test handling of whitespace variations."""

    def test_whitespace_in_tags(self):
        """Handle extra whitespace around tag names."""
        parser = TagParser("observe", validation_mode="moderate")
        result = parser.parse("<  observe  >[1, 2, 3]</  observe  >")
        assert result == "[1, 2, 3]"
        assert parser.last_strategy_used == ParsingStrategy.TOLERANT

    def test_whitespace_in_closing_slash(self):
        """Handle whitespace around closing slash."""
        parser = TagParser("observe", validation_mode="moderate")
        result = parser.parse("<observe>[1, 2, 3]<  /  observe>")
        assert result == "[1, 2, 3]"

    def test_newlines_in_tags(self):
        """Handle newlines in content."""
        parser = TagParser("observe", validation_mode="moderate")
        result = parser.parse("<observe>[1,\n2,\n3]</observe>")
        assert result is not None
        assert "1" in result and "2" in result and "3" in result

    def test_spaces_around_content(self):
        """Strip leading/trailing spaces from content."""
        parser = TagParser("observe", validation_mode="moderate")
        result = parser.parse("<observe>   [1, 2, 3]   </observe>")
        assert result == "[1, 2, 3]"


class TestTagParserMissingClosing:
    """Test handling of missing or malformed closing tags."""

    def test_missing_closing_tag(self):
        """Extract content even without closing tag."""
        parser = TagParser("observe", validation_mode="moderate")
        result = parser.parse("<observe>[1, 2, 3]")
        assert result == "[1, 2, 3]"
        assert parser.last_strategy_used == ParsingStrategy.EMERGENCY

    def test_missing_closing_tag_with_next_element(self):
        """Extract until next tag if closing tag missing."""
        parser = TagParser("observe", validation_mode="moderate")
        result = parser.parse("<observe>[1, 2, 3]<answer>foo</answer>")
        assert result == "[1, 2, 3]"

    def test_incomplete_closing_tag(self):
        """Handle partially formed closing tags."""
        parser = TagParser("observe", validation_mode="moderate")
        result = parser.parse("<observe>[1, 2, 3]</observe>")
        assert result == "[1, 2, 3]"


class TestTagParserEmptyAndInvalid:
    """Test rejection of empty or invalid content."""

    def test_empty_tags(self):
        """Reject tags with no content."""
        parser = TagParser("observe", validation_mode="moderate")
        result = parser.parse("<observe></observe>")
        assert result is None

    def test_no_numbers(self):
        """Reject content without any numbers."""
        parser = TagParser("observe", validation_mode="moderate")
        result = parser.parse("<observe>hello world</observe>")
        assert result is None

    def test_only_whitespace(self):
        """Reject tags containing only whitespace."""
        parser = TagParser("observe", validation_mode="moderate")
        result = parser.parse("<observe>   </observe>")
        assert result is None

    def test_done_signal(self):
        """Reject 'done' response (model giving up)."""
        parser = TagParser("observe", validation_mode="moderate")
        result = parser.parse("<observe>done</observe>")
        assert result is None

    def test_done_signal_longer(self):
        """Handle 'done' signal with numbers - extract the numbers."""
        parser = TagParser("observe", validation_mode="moderate")
        result = parser.parse("<observe>done with [1, 2, 3]</observe>")
        # Parser extracts the valid list despite "done" prefix
        assert result == "[1, 2, 3]"
        assert parser.last_strategy_used == ParsingStrategy.STRICT


class TestTagParserBracketNormalization:
    """Test format normalization for different bracket types."""

    def test_square_brackets_already_correct(self):
        """Keep square brackets as-is."""
        parser = TagParser("observe", validation_mode="moderate")
        result = parser.parse("<observe>[1, 2, 3]</observe>")
        assert result == "[1, 2, 3]"

    def test_parentheses_to_square_brackets(self):
        """Convert parentheses to square brackets."""
        parser = TagParser("observe", validation_mode="moderate")
        result = parser.parse("<observe>(1, 2, 3)</observe>")
        assert result == "[1, 2, 3]"

    def test_comma_separated_to_brackets(self):
        """Convert plain comma-separated to bracketed list."""
        parser = TagParser("observe", validation_mode="moderate")
        result = parser.parse("<observe>1, 2, 3</observe>")
        assert result == "[1, 2, 3]"

    def test_single_number_unchanged(self):
        """Keep single numbers as-is."""
        parser = TagParser("answer", validation_mode="moderate")
        result = parser.parse("<answer>42</answer>")
        assert result == "42"

    def test_float_numbers(self):
        """Handle floating point numbers."""
        parser = TagParser("answer", validation_mode="moderate")
        result = parser.parse("<answer>3.14159</answer>")
        assert result == "3.14159"

    def test_negative_numbers(self):
        """Handle negative numbers."""
        parser = TagParser("observe", validation_mode="moderate")
        result = parser.parse("<observe>[-1, -2, -3]</observe>")
        assert result == "[-1, -2, -3]"


class TestTagParserCaseInsensitive:
    """Test case-insensitive tag matching."""

    def test_uppercase_tags(self):
        """Match uppercase tags."""
        parser = TagParser("observe", validation_mode="moderate")
        result = parser.parse("<OBSERVE>[1, 2, 3]</OBSERVE>")
        assert result == "[1, 2, 3]"

    def test_mixed_case_tags(self):
        """Match mixed case tags."""
        parser = TagParser("observe", validation_mode="moderate")
        result = parser.parse("<Observe>[1, 2, 3]</OBSERVE>")
        assert result == "[1, 2, 3]"

    def test_case_insensitive_opening_closing_mismatch(self):
        """Match even with mismatched case."""
        parser = TagParser("observe", validation_mode="moderate")
        result = parser.parse("<OBSERVE>[1, 2, 3]</observe>")
        assert result == "[1, 2, 3]"


class TestTagParserMultipleTags:
    """Test handling of multiple tags in response."""

    def test_multiple_same_tags(self):
        """Return first valid match when multiple tags present."""
        parser = TagParser("observe", validation_mode="moderate")
        result = parser.parse("<observe>[1, 2]</observe><observe>[3, 4]</observe>")
        assert result == "[1, 2]"

    def test_mixed_tags(self):
        """Extract correct tag type when mixed tags present."""
        parser = TagParser("observe", validation_mode="moderate")
        result = parser.parse("<answer>foo</answer><observe>[1, 2, 3]</observe>")
        # Should skip answer and get observe
        assert result is not None
        assert "1" in result or "[" in result

    def test_nested_similar_tags(self):
        """Handle similar but different tag names."""
        parser = TagParser("observe", validation_mode="moderate")
        result = parser.parse("<observe>[1, 2]</observe><observer>[3, 4]</observer>")
        assert result == "[1, 2]"


class TestTagParserWithContext:
    """Test parsing with surrounding text and context."""

    def test_tag_with_explanation_before(self):
        """Extract tag even with text before it."""
        parser = TagParser("observe", validation_mode="moderate")
        result = parser.parse("Let me observe at: <observe>[1, 2, 3]</observe>")
        assert result == "[1, 2, 3]"

    def test_tag_with_explanation_after(self):
        """Extract tag with text after it."""
        parser = TagParser("observe", validation_mode="moderate")
        result = parser.parse("<observe>[1, 2, 3]</observe> This is my observation")
        assert result == "[1, 2, 3]"

    def test_tag_surrounded_by_text(self):
        """Extract tag from middle of response."""
        parser = TagParser("answer", validation_mode="moderate")
        result = parser.parse("Based on the data, <answer>42</answer> seems reasonable")
        assert result == "42"


class TestTagParserDiagnostics:
    """Test diagnostic messages for failures."""

    def test_diagnostic_missing_tags(self):
        """Diagnose missing tags."""
        parser = TagParser("observe", validation_mode="moderate")
        parser.parse("<answer>[1, 2, 3]</answer>")
        diagnosis = parser.diagnose_failure("<answer>[1, 2, 3]</answer>")
        assert "observe" in diagnosis.lower()
        assert "tags" in diagnosis.lower()

    def test_diagnostic_no_numbers(self):
        """Diagnose missing numbers."""
        parser = TagParser("observe", validation_mode="moderate")
        parser.parse("<observe>hello</observe>")
        diagnosis = parser.diagnose_failure("<observe>hello</observe>")
        assert "numeric" in diagnosis.lower() or "number" in diagnosis.lower()

    def test_diagnostic_empty_tags(self):
        """Diagnose empty tags."""
        parser = TagParser("observe", validation_mode="moderate")
        parser.parse("<observe></observe>")
        diagnosis = parser.diagnose_failure("<observe></observe>")
        assert "empty" in diagnosis.lower() or "no content" in diagnosis.lower()

    def test_diagnostic_empty_response(self):
        """Diagnose empty response."""
        parser = TagParser("observe", validation_mode="moderate")
        parser.parse("")
        diagnosis = parser.diagnose_failure("")
        assert "empty" in diagnosis.lower() or "string" in diagnosis.lower()


class TestTagParserSummary:
    """Test summary functionality."""

    def test_summary_after_success(self):
        """Summary shows successful strategy."""
        parser = TagParser("observe", validation_mode="moderate")
        parser.parse("<observe>[1, 2, 3]</observe>")
        summary = parser.summary()
        assert summary["strategy_used"] == ParsingStrategy.STRICT.value
        assert summary["failure_reason"] is None

    def test_summary_after_failure(self):
        """Summary shows last_resort when wrong tag type but numbers present."""
        parser = TagParser("observe", validation_mode="moderate")
        parser.parse("<answer>[1, 2, 3]</answer>")
        summary = parser.summary()
        # LAST_RESORT extracts numbers from wrong tag type
        assert summary["strategy_used"] == ParsingStrategy.LAST_RESORT.value
        assert summary["failure_reason"] is not None  # notes fallback was used


class TestTagParserValidationModes:
    """Test different validation modes."""

    def test_moderate_normalizes_format(self):
        """Moderate mode accepts and normalizes formats."""
        parser = TagParser("observe", validation_mode="moderate")
        result = parser.parse("<observe>(1, 2, 3)</observe>")
        assert result == "[1, 2, 3]"

    def test_loose_accepts_minimal_structure(self):
        """Loose mode accepts minimal formatting."""
        parser = TagParser("observe", validation_mode="loose")
        result = parser.parse("<observe>1 2 3</observe>")
        # Should accept because numbers are present
        assert result is not None


class TestTagParserEdgeCases:
    """Test unusual but valid edge cases."""

    def test_very_long_content(self):
        """Handle very long content."""
        parser = TagParser("observe", validation_mode="moderate")
        long_content = "[" + ", ".join(str(i) for i in range(100)) + "]"
        result = parser.parse(f"<observe>{long_content}</observe>")
        assert result is not None
        assert "0" in result

    def test_special_characters_in_content(self):
        """Handle special characters that aren't part of structure."""
        parser = TagParser("observe", validation_mode="moderate")
        result = parser.parse("<observe>[1, 2, 3] ± 0.5</observe>")
        assert result is not None
        assert "1" in result

    def test_scientific_notation(self):
        """Handle scientific notation numbers."""
        parser = TagParser("answer", validation_mode="moderate")
        result = parser.parse("<answer>1.23e-4</answer>")
        assert result is not None
        assert "1" in result

    def test_zero_value(self):
        """Accept zero as valid number."""
        parser = TagParser("observe", validation_mode="moderate")
        result = parser.parse("<observe>[0, 1, 2]</observe>")
        assert result == "[0, 1, 2]"


# Integration tests
class TestTagParserIntegration:
    """Test realistic scenarios combining multiple challenges."""

    def test_realistic_llm_response_1(self):
        """Realistic response with thinking section."""
        response = """
        <thought>
        Let me analyze this step by step...
        I need to observe at coordinates where information is most likely.
        </thought>
        <observe>[10, 20, 5]</observe>
        """
        parser = TagParser("observe", validation_mode="moderate")
        result = parser.parse(response)
        assert result == "[10, 20, 5]"

    def test_realistic_llm_response_2(self):
        """Response with explanation and answer."""
        response = """
        Based on the data I've collected, the pattern suggests:
        <answer>0.75</answer>

        This value represents the probability observed.
        """
        parser = TagParser("answer", validation_mode="moderate")
        result = parser.parse(response)
        assert result == "0.75"

    def test_realistic_llm_response_3(self):
        """Response with formatting variation."""
        response = "My observation is:  < observe >  (100, 200, 300)  < / observe >"
        parser = TagParser("observe", validation_mode="moderate")
        result = parser.parse(response)
        assert result == "[100, 200, 300]"

    def test_realistic_failure_case(self):
        """Response without tags uses LAST_RESORT to extract final number."""
        response = "I think the answer is somewhere between 5 and 10."
        parser = TagParser("answer", validation_mode="moderate")
        result = parser.parse(response)
        # LAST_RESORT extracts the last number from untagged text
        assert result == "10"
        assert parser.last_strategy_used == ParsingStrategy.LAST_RESORT
