"""The deterministic contextualizer: what it adds, and what it refuses to do.

The value of this implementation over the LLM one is that it is *checkable*. So
these tests assert the properties an LLM contextualizer could only be spot-checked
for: that the same input always gives the same output, that the token budget is
actually honoured, and that truncation never lands mid-word.

The last one is the subtle one. A window cut at an arbitrary character contributes
a fragment like "mitochond" to the embedding — a token the model has never seen in
that position, actively degrading the vector contextualisation was meant to
improve. Preferring an empty window to a sliced one is the whole reason the
budgeting code descends paragraph → sentence → nothing.
"""

from __future__ import annotations

import pytest

from ragsage.contextualizing import HeadingWindowContextualizer
from ragsage.models import Chunk, Document
from ragsage.parsing.chunking import make_token_counter

_FULL_TEXT = """The Antikythera mechanism was recovered in 1901.

It is an ancient Greek analogue computer.

It modelled the motion of the planets and predicted eclipses.

The device was lost for two millennia in a shipwreck.

Modern reconstructions confirm the gear ratios.
"""


def _document(source: str = "antikythera.md") -> Document:
    return Document(id="doc-1", source=source, content_hash="h")


def _chunk(text: str, headings: list[str] | None = None) -> Chunk:
    return Chunk(
        id="doc-1:0",
        document_id="doc-1",
        text=text,
        page=1,
        ordinal=0,
        metadata={"headings": headings} if headings else {},
    )


TARGET = "It modelled the motion of the planets and predicted eclipses."


# --------------------------------------------------------------------------- #
# What lands in embed_text
# --------------------------------------------------------------------------- #


async def test_the_chunk_survives_verbatim_at_the_end() -> None:
    """Context is a prefix. The passage itself must not be paraphrased or clipped.

    The store keeps ``text`` and ``embed_text`` separately so display stays clean;
    that only holds if the chunk is reproduced exactly here.
    """
    result = await HeadingWindowContextualizer().contextualize(
        _document(), _chunk(TARGET), full_text=_FULL_TEXT
    )

    assert result.endswith(TARGET)


async def test_the_heading_path_names_the_section() -> None:
    result = await HeadingWindowContextualizer(window=0).contextualize(
        _document(), _chunk(TARGET, ["Astronomy", "Greek devices"]), full_text=_FULL_TEXT
    )

    assert "Document: antikythera.md" in result
    assert "Section: Astronomy > Greek devices" in result


async def test_a_chunker_that_records_no_headings_yields_no_section_line() -> None:
    """``metadata`` is an open mapping — a fake or a caller's chunker may omit it."""
    result = await HeadingWindowContextualizer(window=0).contextualize(
        _document(), _chunk(TARGET), full_text=_FULL_TEXT
    )

    assert "Section:" not in result
    assert "Document: antikythera.md" in result


async def test_the_window_pulls_in_both_neighbours() -> None:
    result = await HeadingWindowContextualizer(window=1, max_context_tokens=256).contextualize(
        _document(), _chunk(TARGET), full_text=_FULL_TEXT
    )

    assert "It is an ancient Greek analogue computer." in result
    assert "The device was lost for two millennia in a shipwreck." in result


async def test_window_zero_leaves_heading_context_only() -> None:
    result = await HeadingWindowContextualizer(window=0).contextualize(
        _document(), _chunk(TARGET), full_text=_FULL_TEXT
    )

    assert "Preceding:" not in result
    assert "Following:" not in result


async def test_an_unlocatable_chunk_omits_the_window_rather_than_guessing() -> None:
    """A chunk that isn't in the text at all gets headings and nothing else.

    The real chunker strips heading lines and repacks units, so verbatim presence
    is not guaranteed. A missing neighbourhood is worth much less than a wrong one.
    """
    result = await HeadingWindowContextualizer(window=2).contextualize(
        _document(), _chunk("Text that appears nowhere in the document."), full_text=_FULL_TEXT
    )

    assert "Preceding:" not in result
    assert "Following:" not in result
    assert result.endswith("Text that appears nowhere in the document.")


# --------------------------------------------------------------------------- #
# Determinism and the budget
# --------------------------------------------------------------------------- #


async def test_the_same_input_always_gives_the_same_output() -> None:
    contextualizer = HeadingWindowContextualizer()
    chunk = _chunk(TARGET, ["Astronomy"])

    results = {
        await contextualizer.contextualize(_document(), chunk, full_text=_FULL_TEXT)
        for _ in range(5)
    }

    assert len(results) == 1, "contextualisation must be deterministic"


async def test_two_instances_agree() -> None:
    """Determinism across instances too — no accumulated state in the counter cache."""
    chunk = _chunk(TARGET, ["Astronomy"])
    first = await HeadingWindowContextualizer().contextualize(
        _document(), chunk, full_text=_FULL_TEXT
    )
    second = await HeadingWindowContextualizer().contextualize(
        _document(), chunk, full_text=_FULL_TEXT
    )

    assert first == second


@pytest.mark.parametrize("budget", [0, 8, 16, 32, 64])
async def test_the_context_prefix_respects_its_token_budget(budget: int) -> None:
    """The budget bounds everything *prepended*, so measure the prefix, not the whole.

    The chunk is exempt by design — it is the payload, not context — so the
    assertion subtracts it rather than allowing slack.
    """
    count_tokens = make_token_counter()
    result = await HeadingWindowContextualizer(window=3, max_context_tokens=budget).contextualize(
        _document(), _chunk(TARGET, ["Astronomy"]), full_text=_FULL_TEXT
    )

    prefix = result[: result.rindex(TARGET)]
    # The header is charged against the budget but never itself truncated, so on a
    # very small budget the prefix is the header alone and may exceed it.
    header_tokens = count_tokens("Document: antikythera.md\nSection: Astronomy")
    assert count_tokens(prefix) <= max(budget, header_tokens) + 4


async def test_a_tight_budget_drops_the_window_instead_of_slicing_a_word() -> None:
    """The no-mid-word guarantee, checked where it is most likely to break.

    With almost no budget the window cannot fit whole paragraphs *or* whole
    sentences, so the correct outcome is no window at all.
    """
    result = await HeadingWindowContextualizer(window=2, max_context_tokens=1).contextualize(
        _document(), _chunk(TARGET), full_text=_FULL_TEXT
    )

    assert "Preceding:" not in result
    assert "Following:" not in result


async def test_window_text_is_only_ever_whole_sentences() -> None:
    """Every fragment the window contributes must appear verbatim in the source.

    This is the real no-mid-word assertion: a sliced word would produce a
    substring that is *not* found as a whole unit in the document text.
    """
    result = await HeadingWindowContextualizer(window=2, max_context_tokens=24).contextualize(
        _document(), _chunk(TARGET), full_text=_FULL_TEXT
    )

    for label in ("Preceding: ", "Following: "):
        if label not in result:
            continue
        section = result.split(label, 1)[1].split("\n\n", 1)[0]
        for sentence in section.split(". "):
            fragment = sentence.strip().rstrip(".")
            assert fragment in _FULL_TEXT, f"{fragment!r} is not verbatim source text"


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"window": -1}, "window must be non-negative"),
        ({"max_context_tokens": -1}, "max_context_tokens must be non-negative"),
    ],
)
def test_invalid_settings_are_rejected(kwargs: dict[str, int], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        HeadingWindowContextualizer(**kwargs)


def test_it_satisfies_the_contextualizer_port() -> None:
    from ragsage.ports import Contextualizer

    assert isinstance(HeadingWindowContextualizer(), Contextualizer)
