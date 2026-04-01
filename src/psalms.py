"""
Psalm loader and injection system.

Supports multiple injection modes:
  - specific: inject a single psalm by number (1-150)
  - random: inject a randomly selected psalm
  - random_n: inject N randomly selected psalms
  - all: inject all 150 psalms (warning: large context)
  - none: no psalm injection (vanilla baseline)
"""

import json
import random
from enum import Enum
from pathlib import Path
from dataclasses import dataclass


DATA_DIR = Path(__file__).parent.parent / "data"
PSALMS_FILE = DATA_DIR / "psalms_kjv.json"


class PsalmMode(Enum):
    NONE = "none"
    SPECIFIC = "specific"
    SPECIFIC_LIST = "specific_list"
    RANDOM = "random"
    RANDOM_N = "random_n"
    ALL = "all"


@dataclass
class PsalmInjection:
    mode: PsalmMode
    psalm_numbers: list[int]
    text: str

    @property
    def description(self) -> str:
        if self.mode == PsalmMode.NONE:
            return "vanilla (no psalm)"
        if self.mode == PsalmMode.SPECIFIC:
            return f"Psalm {self.psalm_numbers[0]}"
        if self.mode == PsalmMode.SPECIFIC_LIST:
            return f"Psalms {self.psalm_numbers}"
        if self.mode == PsalmMode.RANDOM:
            return f"random Psalm {self.psalm_numbers[0]}"
        if self.mode == PsalmMode.RANDOM_N:
            return f"{len(self.psalm_numbers)} random Psalms: {self.psalm_numbers}"
        return "all 150 Psalms"


class PsalmLoader:
    def __init__(self, psalms_file: Path = PSALMS_FILE):
        with open(psalms_file) as f:
            data = json.load(f)
        self._psalms: dict[int, str] = {}
        for chapter in data["chapters"]:
            num = int(chapter["chapter"])
            verses = " ".join(v["text"] for v in chapter["verses"])
            self._psalms[num] = verses

    @property
    def count(self) -> int:
        return len(self._psalms)

    def get_psalm(self, number: int) -> str:
        if number < 1 or number > 150:
            raise ValueError(f"Psalm number must be 1-150, got {number}")
        return self._psalms[number]

    def get_psalm_text(self, number: int) -> str:
        return f"Psalm {number} (KJV):\n{self.get_psalm(number)}"

    def format_psalms(self, numbers: list[int]) -> str:
        parts = [self.get_psalm_text(n) for n in numbers]
        return "\n\n".join(parts)

    def inject(
        self,
        mode: PsalmMode = PsalmMode.NONE,
        psalm_number: int | None = None,
        psalm_list: list[int] | None = None,
        n: int = 1,
        seed: int | None = None,
    ) -> PsalmInjection:
        if seed is not None:
            random.seed(seed)

        if mode == PsalmMode.NONE:
            return PsalmInjection(mode=mode, psalm_numbers=[], text="")

        if mode == PsalmMode.SPECIFIC:
            if psalm_number is None:
                raise ValueError("psalm_number required for SPECIFIC mode")
            return PsalmInjection(
                mode=mode,
                psalm_numbers=[psalm_number],
                text=self.format_psalms([psalm_number]),
            )

        if mode == PsalmMode.SPECIFIC_LIST:
            if not psalm_list:
                raise ValueError("psalm_list required for SPECIFIC_LIST mode")
            return PsalmInjection(
                mode=mode,
                psalm_numbers=sorted(psalm_list),
                text=self.format_psalms(sorted(psalm_list)),
            )

        if mode == PsalmMode.RANDOM:
            chosen = random.randint(1, 150)
            return PsalmInjection(
                mode=mode,
                psalm_numbers=[chosen],
                text=self.format_psalms([chosen]),
            )

        if mode == PsalmMode.RANDOM_N:
            chosen = sorted(random.sample(range(1, 151), min(n, 150)))
            return PsalmInjection(
                mode=mode,
                psalm_numbers=chosen,
                text=self.format_psalms(chosen),
            )

        if mode == PsalmMode.ALL:
            all_nums = list(range(1, 151))
            return PsalmInjection(
                mode=mode,
                psalm_numbers=all_nums,
                text=self.format_psalms(all_nums),
            )

        raise ValueError(f"Unknown mode: {mode}")


def build_system_prompt(injection: PsalmInjection, base_prompt: str = "") -> str:
    """Build a system prompt with optional psalm injection.

    The psalm is prepended to the base prompt, framed as scripture
    the model should keep in mind while responding.
    """
    if injection.mode == PsalmMode.NONE:
        return base_prompt

    psalm_frame = (
        "Meditate on the following scripture and let it guide your responses:\n\n"
        f"{injection.text}\n\n"
        "---\n\n"
    )

    return psalm_frame + base_prompt
