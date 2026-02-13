"""Parse and validate XML-like tags from LLM responses."""

import re
import unicodedata
from enum import Enum

# matches numbers: integers, floats, scientific notation (e.g. 1.5e-10)
# \b prevents matching trailing periods in sentences
NUMBER_PATTERN = r"-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?\b"

# minimum content length before checking for "done" (model gave up)
# short responses like "done" shouldn't trigger normal extraction
MIN_CONTENT_LENGTH_FOR_DONE_CHECK = 20

# pre-compiled regex patterns for performance (avoid re-compiling on each call)
_NUMBER_RE = re.compile(NUMBER_PATTERN)
_WELL_FORMED_RE = re.compile(r"\[[\d\s,\.\-]+\]")
_BRACKET_START_RE = re.compile(r"^\s*\w{0,20}\s*\[([\d\s,\.\-]+)\]")
_BRACKET_END_RE = re.compile(r"\[([\d\s,\.\-]+)\]\s*\.?\s*$")
_PAREN_START_RE = re.compile(r"^\s*\w{0,20}\s*\(([\d\s,\.\-]+)\)")
_PAREN_END_RE = re.compile(r"\(([\d\s,\.\-]+)\)\s*\.?\s*$")
_LIST_PATTERN_RE = re.compile(rf"{NUMBER_PATTERN}\s*,\s*{NUMBER_PATTERN}")
_ANSWER_IS_RE = re.compile(rf"answer\s+(?:is|:|=)\s*({NUMBER_PATTERN})", re.IGNORECASE)
_WILL_USE_RE = re.compile(
    rf"(?:I'll|I will)\s+(?:go with|use|choose|answer|predict)\s+({NUMBER_PATTERN})", re.IGNORECASE
)
_THEREFORE_RE = re.compile(
    rf"(?:so|therefore|thus),?\s*({NUMBER_PATTERN})\s*\.?\s*$", re.IGNORECASE
)
_SENTENCE_SPLIT_RE = re.compile(r"[.!?]\s+")
_HAS_DIGIT_RE = re.compile(r"\d")
_NEXT_TAG_RE = re.compile(r"<")


class ValidationMode(Enum):
    STRICT = "strict"  # reject formatting issues
    MODERATE = "moderate"  # normalize format
    LOOSE = "loose"  # accept anything with numbers


class ParsingStrategy(Enum):
    STRICT = "strict_matching"
    TOLERANT = "tolerant_matching"
    EMERGENCY = "emergency_extraction"
    LAST_RESORT = "last_resort_fallback"  # tagless numeric extraction
    FAILED = "all_strategies_failed"


class TagParser:
    """Extract and validate content from XML-like tags in LLM responses.

    Handles common issues: extra whitespace, missing tags, bracket mismatches,
    empty content, and case variations.
    """

    def __init__(self, tag_name: str, validation_mode: str = "moderate"):
        self.tag_name = tag_name.lower()
        self.validation_mode = ValidationMode(validation_mode)
        self.last_strategy_used = None
        self.last_failure_reason = None
        # cache compiled tag-specific patterns
        self._strict_re = re.compile(
            rf"<{self.tag_name}>(.*?)</{self.tag_name}>", re.DOTALL | re.IGNORECASE
        )
        self._tolerant_re = re.compile(
            rf"<\s*{self.tag_name}\s*>(.*?)<\s*/\s*{self.tag_name}\s*>", re.DOTALL | re.IGNORECASE
        )
        self._opening_re = re.compile(rf"<\s*{self.tag_name}\s*>", re.IGNORECASE)
        self._closing_re = re.compile(rf"<\s*/\s*{self.tag_name}\s*>", re.IGNORECASE)
        self._answer_tag_re = re.compile(rf"<answer>\s*({NUMBER_PATTERN})", re.IGNORECASE)
        self._observe_tag_re = re.compile(rf"<observe>\s*({NUMBER_PATTERN})", re.IGNORECASE)

    def _normalize_text(self, text: str) -> str:
        """Normalize Unicode text (fullwidth digits, punctuation) to ASCII equivalents."""
        if not text:
            return ""
        # NFKC normalization converts fullwidth digits (１２３) and punctuation (，：) to ASCII
        return unicodedata.normalize("NFKC", text)

    def parse(self, response: str) -> str | None:
        """Extract and validate tag content using hierarchical fallback strategies."""
        if not response or not isinstance(response, str):
            self.last_strategy_used = ParsingStrategy.FAILED
            self.last_failure_reason = "Empty or non-string response"
            return None

        # Normalize Unicode (handles fullwidth digits from Chinese models like glm-4.7, kimi-k2)
        response = self._normalize_text(response)

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

        # LAST_RESORT: tagless numeric extraction (only in LOOSE or MODERATE mode)
        if self.validation_mode != ValidationMode.STRICT:
            result = self._extract_from_reasoning(response)
            if result is not None:
                self.last_strategy_used = ParsingStrategy.LAST_RESORT
                self.last_failure_reason = "Used tagless fallback - no valid tags found"
                return result

        self.last_strategy_used = ParsingStrategy.FAILED
        self.last_failure_reason = "No valid tags found after trying all strategies"
        return None

    def _parse_strict(self, response: str) -> str | None:
        matches = self._strict_re.findall(response)

        for match in matches:
            content = self._validate_and_normalize(match)
            if content:
                return content

        return None

    def _parse_tolerant(self, response: str) -> str | None:
        """Allow whitespace around tags and slashes."""
        matches = self._tolerant_re.findall(response)

        for match in matches:
            content = self._validate_and_normalize(match)
            if content:
                return content

        return None

    def _parse_emergency(self, response: str) -> str | None:
        """Lenient extraction for malformed tags (missing closing tag, etc)."""
        opening_match = self._opening_re.search(response)

        if not opening_match:
            self.last_failure_reason = f"No opening <{self.tag_name}> tag found"
            return None

        start_pos = opening_match.end()

        closing_match = self._closing_re.search(response[start_pos:])

        if closing_match:
            end_pos = start_pos + closing_match.start()
        else:
            next_tag_match = _NEXT_TAG_RE.search(response[start_pos:])
            if next_tag_match:
                end_pos = start_pos + next_tag_match.start()
            else:
                end_pos = len(response)

        content = response[start_pos:end_pos].strip()
        return self._validate_and_normalize(content)

    def _validate_and_normalize(self, content: str) -> str | None:
        """Validate content has numbers and normalize bracket format."""
        content = content.strip()
        # Normalize Unicode in content as well
        content = self._normalize_text(content)

        if not content:
            self.last_failure_reason = "Tag contains no content"
            return None

        # Tightened "done" filter: exact match only (was rejecting valid outputs like "done: 5")
        if re.fullmatch(r"\s*done\s*\.?\s*", content, flags=re.IGNORECASE):
            self.last_failure_reason = "Content is just 'done' (model gave up)"
            return None

        numbers = _NUMBER_RE.findall(content)
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
        return bool(_WELL_FORMED_RE.search(content))

    def _normalize_format(self, content: str) -> str:
        """Normalize brackets and extract answers from verbose reasoning."""
        content = content.strip()

        # extract square brackets if present
        if "[" in content and "]" in content:
            match = _BRACKET_START_RE.search(content)
            if match:
                return f"[{match.group(1)}]"
            match = _BRACKET_END_RE.search(content)
            if match:
                return f"[{match.group(1)}]"

        # convert parentheses to brackets (avoid matching function notation like P(0))
        if "(" in content and ")" in content:
            match = _PAREN_START_RE.search(content)
            if match and "," in match.group(1):
                return f"[{match.group(1)}]"
            match = _PAREN_END_RE.search(content)
            if match and "," in match.group(1):
                return f"[{match.group(1)}]"

        # handle verbose reasoning model output
        if len(content) > 100:
            extracted = self._extract_from_reasoning(content)
            if extracted:
                return extracted

        # extract numbers
        numbers = _NUMBER_RE.findall(content)
        if len(numbers) > 1:
            if _LIST_PATTERN_RE.search(content):
                return f"[{', '.join(numbers)}]"
            else:
                return numbers[-1]
        elif len(numbers) == 1:
            return numbers[0]

        return content

    def _extract_from_reasoning(self, reasoning_text: str) -> str | None:
        """Extract final answer from verbose reasoning (for o1, DeepSeek-Reasoner, etc)."""
        text = reasoning_text.strip()

        # look for explicit answer patterns (use cached compiled patterns)
        for tag_re in [self._answer_tag_re, self._observe_tag_re]:
            matches = tag_re.findall(text)
            if matches:
                return matches[-1]

        match = _ANSWER_IS_RE.search(text)
        if match:
            return match.group(1)

        match = _WILL_USE_RE.search(text)
        if match:
            return match.group(1)

        match = _THEREFORE_RE.search(text)
        if match:
            return match.group(1)

        # extract last number from last sentence
        sentences = _SENTENCE_SPLIT_RE.split(text)
        if sentences:
            last_sentence = sentences[-1]
            numbers = _NUMBER_RE.findall(last_sentence)
            if numbers:
                return numbers[-1]

        # fallback: last number overall (prefer integers)
        all_numbers = _NUMBER_RE.findall(text)
        if all_numbers:
            integers = [n for n in all_numbers if "." not in n]
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

        if not self._opening_re.search(response):
            return (
                f"Opening <{self.tag_name}> tag is malformed or missing. "
                f"Use proper format: <{self.tag_name}>content</{self.tag_name}>"
            )

        matches = self._tolerant_re.findall(response)
        if matches and all(not m.strip() for m in matches):
            return f"Tags are empty - no content between <{self.tag_name}> and </{self.tag_name}>"

        if not _HAS_DIGIT_RE.search(response):
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
