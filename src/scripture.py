"""
Generalized scripture loader and injection system.

Supports loading any biblical book from the aruljohn/Bible-kjv JSON format
and injecting chapter selections into system prompts.

Reuses the same injection modes as the psalm system:
  - specific: inject a single chapter by number
  - specific_list: inject a list of chapters by number
  - random: inject a randomly selected chapter
  - random_n: inject N randomly selected chapters
  - all: inject all chapters
  - none: no injection (vanilla baseline)
"""

import json
import random
from pathlib import Path
from dataclasses import dataclass

from .psalms import PsalmMode  # reuse the same enum


DATA_DIR = Path(__file__).parent.parent / "data"


@dataclass
class ScriptureInjection:
    mode: PsalmMode
    book: str
    chapter_numbers: list[int]
    text: str

    @property
    def description(self) -> str:
        if self.mode == PsalmMode.NONE:
            return f"vanilla (no {self.book.lower()})"
        if self.mode == PsalmMode.SPECIFIC:
            return f"{self.book} {self.chapter_numbers[0]}"
        if self.mode == PsalmMode.SPECIFIC_LIST:
            return f"{self.book} {self.chapter_numbers}"
        if self.mode == PsalmMode.RANDOM:
            return f"random {self.book} {self.chapter_numbers[0]}"
        if self.mode == PsalmMode.RANDOM_N:
            return f"{len(self.chapter_numbers)} random {self.book}: {self.chapter_numbers}"
        return f"all {self.book}"


class ScriptureLoader:
    def __init__(self, book_name: str, json_file: Path | None = None):
        self.book_name = book_name
        if json_file is None:
            json_file = DATA_DIR / f"{book_name.lower()}_kjv.json"
        with open(json_file) as f:
            data = json.load(f)
        self._chapters: dict[int, str] = {}
        for chapter in data["chapters"]:
            num = int(chapter["chapter"])
            verses = " ".join(v["text"] for v in chapter["verses"])
            self._chapters[num] = verses
        self._max_chapter = max(self._chapters.keys())

    @property
    def count(self) -> int:
        return len(self._chapters)

    def get_chapter(self, number: int) -> str:
        if number not in self._chapters:
            raise ValueError(
                f"{self.book_name} chapter {number} not found. "
                f"Available: 1-{self._max_chapter}"
            )
        return self._chapters[number]

    def get_chapter_text(self, number: int) -> str:
        return f"{self.book_name} {number} (KJV):\n{self.get_chapter(number)}"

    def format_chapters(self, numbers: list[int]) -> str:
        parts = [self.get_chapter_text(n) for n in numbers]
        return "\n\n".join(parts)

    def inject(
        self,
        mode: PsalmMode = PsalmMode.NONE,
        chapter_number: int | None = None,
        chapter_list: list[int] | None = None,
        n: int = 1,
        seed: int | None = None,
    ) -> ScriptureInjection:
        if seed is not None:
            random.seed(seed)

        if mode == PsalmMode.NONE:
            return ScriptureInjection(
                mode=mode, book=self.book_name, chapter_numbers=[], text=""
            )

        if mode == PsalmMode.SPECIFIC:
            if chapter_number is None:
                raise ValueError("chapter_number required for SPECIFIC mode")
            return ScriptureInjection(
                mode=mode,
                book=self.book_name,
                chapter_numbers=[chapter_number],
                text=self.format_chapters([chapter_number]),
            )

        if mode == PsalmMode.SPECIFIC_LIST:
            if not chapter_list:
                raise ValueError("chapter_list required for SPECIFIC_LIST mode")
            return ScriptureInjection(
                mode=mode,
                book=self.book_name,
                chapter_numbers=sorted(chapter_list),
                text=self.format_chapters(sorted(chapter_list)),
            )

        if mode == PsalmMode.RANDOM:
            chosen = random.choice(list(self._chapters.keys()))
            return ScriptureInjection(
                mode=mode,
                book=self.book_name,
                chapter_numbers=[chosen],
                text=self.format_chapters([chosen]),
            )

        if mode == PsalmMode.RANDOM_N:
            all_nums = list(self._chapters.keys())
            chosen = sorted(random.sample(all_nums, min(n, len(all_nums))))
            return ScriptureInjection(
                mode=mode,
                book=self.book_name,
                chapter_numbers=chosen,
                text=self.format_chapters(chosen),
            )

        if mode == PsalmMode.ALL:
            all_nums = sorted(self._chapters.keys())
            return ScriptureInjection(
                mode=mode,
                book=self.book_name,
                chapter_numbers=all_nums,
                text=self.format_chapters(all_nums),
            )

        raise ValueError(f"Unknown mode: {mode}")


def build_scripture_system_prompt(
    injection: ScriptureInjection, base_prompt: str = ""
) -> str:
    """Build a system prompt with optional scripture injection."""
    if injection.mode == PsalmMode.NONE:
        return base_prompt

    scripture_frame = (
        "Meditate on the following scripture and let it guide your responses:\n\n"
        f"{injection.text}\n\n"
        "---\n\n"
    )

    return scripture_frame + base_prompt
