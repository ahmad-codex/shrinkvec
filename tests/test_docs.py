"""The examples in docs/python.md run as written, so the page cannot drift from the code."""

import re
from pathlib import Path

PAGE = Path(__file__).parent.parent / "docs" / "python.md"
SKIP = "<!-- not tested"


def python_blocks(text: str) -> list[str]:
    """The ```python blocks in order, leaving out any right after a "not tested" comment."""
    blocks = []
    for m in re.finditer(r"^```python\n(.*?)^```$", text, re.MULTILINE | re.DOTALL):
        before = text[: m.start()].rstrip().splitlines()
        if before and before[-1].startswith(SKIP):
            continue
        blocks.append(m.group(1))
    return blocks


def test_python_api_page_runs():
    blocks = python_blocks(PAGE.read_text(encoding="utf-8"))
    assert len(blocks) >= 4
    namespace: dict = {}
    for i, block in enumerate(blocks):
        exec(compile(block, f"{PAGE.name} block {i + 1}", "exec"), namespace)


def test_untested_blocks_still_compile():
    text = PAGE.read_text(encoding="utf-8")
    every = re.findall(r"^```python\n(.*?)^```$", text, re.MULTILINE | re.DOTALL)
    assert len(every) > len(python_blocks(text))
    for block in every:
        compile(block, PAGE.name, "exec")
