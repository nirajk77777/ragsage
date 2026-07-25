"""The conversational path: replying to messages that were never corpus questions.

Two things are under test and they pull in opposite directions. A greeting must
get a greeting — telling a user that "hi" is not in their documents is a lie
about what the engine did. And a real question must *still* be retrieved, cited,
and honestly refused when the corpus can't support it, even when it opens with a
greeting. The classifier is deliberately narrow so the second guarantee wins
every tie.
"""

from __future__ import annotations

import pytest
from ragsage.fakes import FakeEngineKit, FakeLLMClient
from ragsage.models import AnswerComplete, AnswerToken, Outcome, Usage
from ragsage.query import NOT_FOUND_MESSAGE
from ragsage.smalltalk import is_small_talk, normalise

from ragsage import IngestionPipeline, QueryEngine, RawSource, Scope

BIOLOGY = b"The mitochondrion is the powerhouse of the cell and produces ATP."


# --------------------------------------------------------------------------- #
# The classifier — pure, so tested directly
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "message",
    [
        "hi",
        "Hi!",
        "  hello  ",
        "Hey there",
        "Good morning",
        "how's it going?",
        "Thanks!",
        "thank you so much",
        "What can you do?",
        "who are you",
        "bye",
    ],
)
def test_conversational_messages_are_recognised(message: str) -> None:
    assert is_small_talk(message) is True


@pytest.mark.parametrize(
    "message",
    [
        # Real questions, including ones that open politely. A greeting *prefix*
        # must never route a genuine question away from the corpus.
        "hi, what does the contract say about termination?",
        "hello can you summarise the report",
        "thanks — now what is ATP?",
        "what can you do to reduce the churn rate described in the deck?",
        "who are you calling in the escalation policy?",
        "What is the capital of France?",
        "help me understand section 4",
        "",
        "   ",
    ],
)
def test_real_questions_are_not_small_talk(message: str) -> None:
    assert is_small_talk(message) is False


def test_normalise_strips_case_punctuation_and_apostrophes() -> None:
    assert normalise("  How's IT going?!  ") == "hows it going"


# --------------------------------------------------------------------------- #
# The engine
# --------------------------------------------------------------------------- #


async def test_greeting_gets_a_conversational_reply_not_a_not_found(
    pipeline: IngestionPipeline, engine: QueryEngine, scope: Scope
) -> None:
    await pipeline.ingest(RawSource(name="bio.txt", content=BIOLOGY), scope)

    answer = await engine.query("hi", scope)

    assert answer.outcome is Outcome.CONVERSATIONAL
    assert answer.text == FakeLLMClient.SMALL_TALK_REPLY
    assert answer.text != NOT_FOUND_MESSAGE
    # Conversational, so nothing was retrieved and nothing may be cited.
    assert answer.citations == ()
    assert answer.grounded is False


async def test_greeting_never_touches_the_corpus(
    engine: QueryEngine, kit: FakeEngineKit, scope: Scope
) -> None:
    await engine.query("hello", scope)

    # No retrieve/rerank spans at all: the short-circuit is before the funnel,
    # which is what makes small talk free rather than merely well-worded.
    assert kit.tracer.find_span("retrieve") is None
    assert kit.tracer.find_span("rerank") is None
    assert kit.tracer.find_span("query").attributes["outcome"] == "conversational"


async def test_real_question_still_gets_the_honest_not_found(
    pipeline: IngestionPipeline, engine: QueryEngine, scope: Scope
) -> None:
    await pipeline.ingest(RawSource(name="bio.txt", content=BIOLOGY), scope)

    answer = await engine.query("Who won the 1998 football world cup?", scope)

    assert answer.outcome is Outcome.NOT_FOUND
    assert answer.text == NOT_FOUND_MESSAGE


async def test_question_opening_with_a_greeting_is_still_answered_from_the_corpus(
    pipeline: IngestionPipeline, engine: QueryEngine, scope: Scope
) -> None:
    await pipeline.ingest(RawSource(name="bio.txt", content=BIOLOGY), scope)

    answer = await engine.query("hi, what does the mitochondrion produce?", scope)

    # The politeness is not the message; the question is. It must be retrieved
    # and cited exactly as if the "hi," were not there.
    assert answer.outcome is Outcome.ANSWERED
    assert "ATP" in answer.text
    assert answer.citations


async def test_small_talk_streams_like_any_other_answer(engine: QueryEngine, scope: Scope) -> None:
    events = [event async for event in engine.stream("thanks", scope)]

    # Same event shape as the grounded path — tokens, a usage, a terminal
    # complete — so a consumer needs no special case for the third outcome.
    assert any(isinstance(e, AnswerToken) for e in events)
    usage = next(e for e in events if isinstance(e, Usage))
    assert usage.sources == 0  # nothing retrieved
    complete = events[-1]
    assert isinstance(complete, AnswerComplete)
    assert complete.outcome is Outcome.CONVERSATIONAL
    assert "".join(e.text for e in events if isinstance(e, AnswerToken)).strip() == complete.text


async def test_buffered_and_streamed_small_talk_agree(engine: QueryEngine, scope: Scope) -> None:
    buffered = await engine.query("hey", scope)
    complete = [e async for e in engine.stream("hey", scope)][-1]

    assert isinstance(complete, AnswerComplete)
    assert buffered.text == complete.text
    assert buffered.outcome is complete.outcome


async def test_model_declining_a_misjudged_message_falls_back_to_not_found(
    kit: FakeEngineKit, scope: Scope
) -> None:
    """The phrase list proposes; the model disposes.

    The small-talk prompt tells the model to reply with the not-found message if
    the message actually asks for information. Honour that: a message the gate
    misjudged lands on the honest refusal, never on an invented answer.
    """

    class DecliningLLM(FakeLLMClient):
        async def generate(self, prompt: str):  # type: ignore[override]
            for word in NOT_FOUND_MESSAGE.split():
                yield word + " "

    engine = QueryEngine(
        embedder=kit.embedder,
        vector_store=kit.vector_store,
        lexical_store=kit.lexical_store,
        reranker=kit.reranker,
        llm=DecliningLLM(),
        tracer=kit.tracer,
    )

    answer = await engine.query("help", scope)

    assert answer.outcome is Outcome.NOT_FOUND
    assert answer.text == NOT_FOUND_MESSAGE
    assert answer.citations == ()
