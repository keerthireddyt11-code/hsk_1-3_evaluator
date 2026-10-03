from typing import List
from pydantic import BaseModel, Field


class Assessment(BaseModel):
    is_correct: bool
    corrected_hanzi: str
    corrected_pinyin: str
    english_translation: str
    explanation: str
    out_of_scope_words: List[str] = Field(default_factory=list)