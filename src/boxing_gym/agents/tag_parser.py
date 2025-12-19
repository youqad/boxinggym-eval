"""Parse and validate XML-like tags from LLM responses."""

import re
from typing import Optional
from enum import Enum


# matches numbers: integers, floats, scientific notation (e.g. 1.5e-10)
# \b prevents matching trailing periods in sentences
NUMBER_PATTERN = r'-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?\b'


class ValidationMode(Enum):
    STRICT = "strict"      # reject formatting issues
    MODERATE = "moderate"  # normalize format
    LOOSE = "loose"        # accept anything with numbers


class ParsingStrategy(Enum):
    STRICT = "strict_matching"
    TOLERANT = "tolerant_matching"
    EMERGENCY = "emergency_extraction"
    FAILED = "all_strategies_failed"


class TagParser:
    """Extract and validate content from XML-like tags in LLM responses.

    Handles common issues: extra whitespace, missing tags, bracket mismatches,
    empty content, and case variations.
    """

    def __init__(
        self,
        tag_name: str,
        validation_mode: str = "moderate"
    ):
        self.tag_name = tag_name.lower()
        self.validation_mode = ValidationMode(validation_mode)
        self.last_strategy_used = None
        self.last_failure_reason = None

    def parse(self, response: str) -> Optional[str]:
        """Extract and validate tag content using hierarchical fallback strategies."""
        if not response or not isinstance(response, str):
            self.last_strategy_used = ParsingStrategy.FAILED
            self.last_failure_reason = "Empty or non-string response"
            return None

        result = self._parse_strict(response)
        if result is not None:
            self.last_strategy_used = ParsingStrategy.STRICT
            return result

        result = self._parse_tolerant(response)
        if result is not None:
            self.last_strategy_used = ParsingStrategy.TOLERANT
            return result

        result = self._parse_emergency(response)
        if result is not None:
            self.last_strategy_used = ParsingStrategy.EMERGENCY
            return result
        self.last_strategy_used = ParsingStrategy.FAILED
        self.last_failure_reason = "No valid tags found after trying all strategies"
        return None

    def _parse_strict(self, response: str) -> Optional[str]:
        pattern = rf'<{self.tag_name}>(.*?)</{self.tag_name}>'
        matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)

        for match in matches:
            content = self._validate_and_normalize(match)
            if content:
                return content

        return None

    def _parse_tolerant(self, response: str) -> Optional[str]:
        """Allow whitespace around tags and slashes."""
        pattern = rf'<\s*{self.tag_name}\s*>(.*?)<\s*/\s*{self.tag_name}\s*>'
        matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)

        for match in matches:
            content = self._validate_and_normalize(match)
            if content:
                return content

        return None

    def _parse_emergency(self, response: str) -> Optional[str]:
        """Lenient extraction for malformed tags (missing closing tag, etc)."""
        opening_pattern = rf'<\s*{self.tag_name}\s*>'
        opening_match = re.search(opening_pattern, response, re.IGNORECASE)

        if not opening_match:
            self.last_failure_reason = f"No opening <{self.tag_name}> tag found"
            return None

        start_pos = opening_match.end()

        closing_pattern = rf'<\s*/\s*{self.tag_name}\s*>'
        closing_match = re.search(closing_pattern, response[start_pos:], re.IGNORECASE)

        if closing_match:
            end_pos = start_pos + closing_match.start()
        else:
            next_tag_match = re.search(r'<', response[start_pos:])
            if next_tag_match:
                end_pos = start_pos + next_tag_match.start()
            else:
                end_pos = len(response)

        content = response[start_pos:end_pos].strip()
        return self._validate_and_normalize(content)

    def _validate_and_normalize(self, content: str) -> Optional[str]:
        """Validate content has numbers and normalize bracket format."""
        content = content.strip()

        if not content:
            self.last_failure_reason = "Tag contains no content"
            return None

        if len(content) < 20 and "done" in content.lower():
            self.last_failure_reason = "Content is just 'done' (model gave up)"
            return None

        numbers = re.findall(NUMBER_PATTERN, content)
        if not numbers:
            self.last_failure_reason = "Content has no numeric values"
            return None
        if self.validation_mode == ValidationMode.STRICT:
            if not self._is_well_formed(content):
                self.last_failure_reason = f"Content format not well-formed: {content}"
                return None
            return content
        else:
            return self._normalize_format(content)

    def _is_well_formed(self, content: str) -> bool:
        return bool(re.search(r'\[[\d\s,\.\-]+\]', content))

    def _normalize_format(self, content: str) -> str:
        """Normalize brackets and extract answers from verbose reasoning."""
        content = content.strip()

        # extract square brackets if present
        if '[' in content and ']' in content:
            match = re.search(r'^\s*\w{0,20}\s*\[([\d\s,\.\-]+)\]', content)
            if match:
                return f"[{match.group(1)}]"
            match = re.search(r'\[([\d\s,\.\-]+)\]\s*\.?\s*$', content)
            if match:
                return f"[{match.group(1)}]"

        # convert parentheses to brackets (avoid matching function notation like P(0))
        if '(' in content and ')' in content:
            match = re.search(r'^\s*\w{0,20}\s*\(([\d\s,\.\-]+)\)', content)
            if match and ',' in match.group(1):
                return f"[{match.group(1)}]"
            match = re.search(r'\(([\d\s,\.\-]+)\)\s*\.?\s*$', content)
            if match and ',' in match.group(1):
                return f"[{match.group(1)}]"

        # handle verbose reasoning model output
        if len(content) > 100:
            extracted = self._extract_from_reasoning(content)
            if extracted:
                return extracted

        # extract numbers
        numbers = re.findall(NUMBER_PATTERN, content)
        if len(numbers) > 1:
            list_pattern = rf'{NUMBER_PATTERN}\s*,\s*{NUMBER_PATTERN}'
            if re.search(list_pattern, content):
                return f"[{', '.join(numbers)}]"
            else:
                return numbers[-1]
        elif len(numbers) == 1:
            return numbers[0]

        return content

    def _extract_from_reasoning(self, reasoning_text: str) -> Optional[str]:
        """Extract final answer from verbose reasoning (for o1, DeepSeek-Reasoner, etc)."""
        text = reasoning_text.strip()

        # look for explicit answer patterns
        answer_patterns = [
            rf'<answer>\s*({NUMBER_PATTERN})',
            rf'<observe>\s*({NUMBER_PATTERN})',
        ]

        for pattern in answer_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                return matches[-1]

        match = re.search(rf'answer\s+(?:is|:|=)\s*({NUMBER_PATTERN})', text, re.IGNORECASE)
        if match:
            return match.group(1)

        match = re.search(rf"(?:I'll|I will)\s+(?:go with|use|choose|answer|predict)\s+({NUMBER_PATTERN})", text, re.IGNORECASE)
        if match:
            return match.group(1)

        match = re.search(rf'(?:so|therefore|thus),?\s*({NUMBER_PATTERN})\s*\.?\s*$', text, re.IGNORECASE)
        if match:
            return match.group(1)

        # extract last number from last sentence
        sentences = re.split(r'[.!?]\s+', text)
        if sentences:
            last_sentence = sentences[-1]
            numbers = re.findall(NUMBER_PATTERN, last_sentence)
            if numbers:
                return numbers[-1]

        # fallback: last number overall (prefer integers)
        all_numbers = re.findall(NUMBER_PATTERN, text)
        if all_numbers:
            integers = [n for n in all_numbers if '.' not in n]
            if integers:
                return integers[-1]
            else:
                return all_numbers[-1]

        return None

    def diagnose_failure(self, response: str) -> str:
        """Generate diagnostic message for parsing failures."""
        if not response or not isinstance(response, str):
            return "Response is empty or not a string"

        if self.tag_name.lower() not in response.lower():
            return (
                f"Response does not contain <{self.tag_name}> tags. "
                f"Wrap your answer in <{self.tag_name}>...</{self.tag_name}> tags."
            )

        opening_pattern = rf'<\s*{self.tag_name}\s*>'
        if not re.search(opening_pattern, response, re.IGNORECASE):
            return (
                f"Opening <{self.tag_name}> tag is malformed or missing. "
                f"Use proper format: <{self.tag_name}>content</{self.tag_name}>"
            )

        pattern = rf'<\s*{self.tag_name}\s*>(.*?)<\s*/\s*{self.tag_name}\s*>'
        matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
        if matches and all(not m.strip() for m in matches):
            return f"Tags are empty - no content between <{self.tag_name}> and </{self.tag_name}>"

        if not re.search(r'\d', response):
            return (
                f"Content has no numeric values. "
                f"You must include numbers in the <{self.tag_name}> tags."
            )

        if self.last_failure_reason:
            return self.last_failure_reason

        return (
            f"Response does not match expected format. "
            f"Use <{self.tag_name}>your_answer</{self.tag_name}> with numeric content."
        )

    def summary(self) -> dict:
        """Get summary of last parsing attempt."""
        return {
            "strategy_used": self.last_strategy_used.value if self.last_strategy_used else None,
            "failure_reason": self.last_failure_reason,
        }
