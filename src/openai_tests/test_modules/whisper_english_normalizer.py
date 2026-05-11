"""English transcript normalization for ASR WER comparison."""

from __future__ import annotations

# Adapted from the English-only behavior of huggingface/open_asr_leaderboard
# normalizer/normalizer.py at 0009f5fe216d63eea809f9849f4d4534c6ab341e.
# Copyright 2022 The OpenAI team and The HuggingFace Team. Licensed under the
# Apache License, Version 2.0.
import re
import unicodedata

CONTRACTIONS = (
  (r"\bwon't\b", "will not"),
  (r"\bcan't\b", "can not"),
  (r"\blet's\b", "let us"),
  (r"\by'all\b", "you all"),
  (r"\bwanna\b", "want to"),
  (r"\bgotta\b", "got to"),
  (r"\bgonna\b", "going to"),
  (r"\bwoulda\b", "would have"),
  (r"\bcoulda\b", "could have"),
  (r"\bshoulda\b", "should have"),
  (r"\bma'am\b", "madam"),
  (r"n't\b", " not"),
  (r"'re\b", " are"),
  (r"'s\b", " is"),
  (r"'d\b", " would"),
  (r"'ll\b", " will"),
  (r"'ve\b", " have"),
  (r"'m\b", " am"),
)

SPELLING_NORMALIZER = {
  "analyse": "analyze",
  "analysed": "analyzed",
  "analyses": "analyzes",
  "analysing": "analyzing",
  "colour": "color",
  "coloured": "colored",
  "colouring": "coloring",
  "colours": "colors",
  "favour": "favor",
  "favours": "favors",
  "honour": "honor",
  "honours": "honors",
  "neighbour": "neighbor",
  "neighbours": "neighbors",
  "organise": "organize",
  "organised": "organized",
  "realise": "realize",
  "realised": "realized",
}

NUMBER_WORDS = {
  "zero": 0,
  "oh": 0,
  "one": 1,
  "two": 2,
  "three": 3,
  "four": 4,
  "five": 5,
  "six": 6,
  "seven": 7,
  "eight": 8,
  "nine": 9,
  "ten": 10,
  "eleven": 11,
  "twelve": 12,
  "thirteen": 13,
  "fourteen": 14,
  "fifteen": 15,
  "sixteen": 16,
  "seventeen": 17,
  "eighteen": 18,
  "nineteen": 19,
  "twenty": 20,
  "thirty": 30,
  "forty": 40,
  "fifty": 50,
  "sixty": 60,
  "seventy": 70,
  "eighty": 80,
  "ninety": 90,
}


class EnglishTextNormalizer:
  """Apply a compact English Whisper-style transcript normalization pipeline."""

  def __call__(self, value: str) -> str:
    """Normalize text for fair word-level ASR comparison."""

    value = normalize_apostrophes(value.lower())
    value = remove_bracketed_text(value)
    value = re.sub(r"\b(hmm|mm|mhm|mmm|uh|um)\b", " ", value)
    value = re.sub(r"\s+'", "'", value)
    for pattern, replacement in CONTRACTIONS:
      value = re.sub(pattern, replacement, value)
    value = re.sub(r"(\d),(\d)", r"\1\2", value)
    value = remove_symbols_and_diacritics(value)
    words = [SPELLING_NORMALIZER.get(word, word) for word in value.split()]
    words = normalize_number_words(words)
    return " ".join(words)


def normalize_apostrophes(value: str) -> str:
  """Map curly apostrophes to ASCII apostrophes before contraction handling."""

  return value.replace("\N{RIGHT SINGLE QUOTATION MARK}", "'").replace("\N{LEFT SINGLE QUOTATION MARK}", "'")


def remove_bracketed_text(value: str) -> str:
  """Drop bracketed and parenthesized annotations from transcripts."""

  value = re.sub(r"[<\[][^>\]]*[>\]]", "", value)
  return re.sub(r"\(([^)]+?)\)", "", value)


def remove_symbols_and_diacritics(value: str) -> str:
  """Remove punctuation and diacritics while preserving word boundaries."""

  normalized = unicodedata.normalize("NFKD", value.replace("-", " "))
  characters = []
  for character in normalized:
    category = unicodedata.category(character)
    if category == "Mn":
      continue
    characters.append(character if character.isalnum() or character.isspace() else " ")
  return re.sub(r"\s+", " ", "".join(characters)).strip()


def normalize_number_words(words: list[str]) -> list[str]:
  """Normalize simple English number words to digit tokens."""

  normalized: list[str] = []
  index = 0
  while index < len(words):
    word = words[index]
    next_word = words[index + 1] if index + 1 < len(words) else None
    if word in NUMBER_WORDS and next_word in NUMBER_WORDS and NUMBER_WORDS[word] >= 20 and NUMBER_WORDS[next_word] < 10:
      normalized.append(str(NUMBER_WORDS[word] + NUMBER_WORDS[next_word]))
      index += 2
      continue
    if word in NUMBER_WORDS and word not in {"one"}:
      normalized.append(str(NUMBER_WORDS[word]))
    else:
      normalized.append(word)
    index += 1
  return normalized
