from typing import List
from pydantic import BaseModel, Field


class Assessment(BaseModel):
    # Field order matters: the model writes the explanation BEFORE it commits to a verdict.
    explanation: str
    is_correct: bool
    error_type: str = ""          # e.g. "word order", "missing measure word", "none"
    corrected_hanzi: str
    corrected_pinyin: str
    english_translation: str
    out_of_scope_words: List[str] = Field(default_factory=list)