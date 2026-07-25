"""Recognising the messages that were never questions about the corpus.

A grounded question-answering engine that answers *only* from retrieved sources
has one embarrassing failure: say "hi" to it and it solemnly reports that your
greeting could not be found in your documents. The message was never a question
about the corpus, so searching it was the wrong thing to do and reporting a miss
is a lie about what happened.

This module is the gate in front of retrieval. It is deliberately **narrow and
deterministic**: a curated set of whole-message phrases, matched after
normalisation, with no model call, no latency, and no cost. Anything it doesn't
recognise falls through to the grounded path unchanged — so the honest not-found
guarantee is untouched for every real question, and the cost of the gate not
firing is only that a greeting gets the old reply.

Matching is *whole-message* on purpose. "hi" is small talk; "hi, what does the
contract say about termination?" is a question that happens to open politely,
and must still be retrieved and cited. A rule that fired on a greeting *prefix*
would quietly route real questions away from the corpus — the one failure mode
worth engineering against here, because it turns a citable answer into a chatty
non-answer.

The gate is a hint, not a verdict: :mod:`ragsage.query` still runs the reply past
the model with a prompt that refuses to state facts, so a false positive degrades
to the honest not-found rather than to an invented answer.
"""

from __future__ import annotations

import re

_PUNCTUATION = re.compile(r"[^a-z\s]+")
_WHITESPACE = re.compile(r"\s+")
_APOSTROPHES = ("'", "\u2019")  # typed, and the curly one phone keyboards substitute

SMALL_TALK: frozenset[str] = frozenset(
    {
        # Greetings.
        "hi",
        "hi there",
        "hello",
        "hello there",
        "hey",
        "hey there",
        "heya",
        "hiya",
        "howdy",
        "yo",
        "greetings",
        "good morning",
        "good afternoon",
        "good evening",
        "good day",
        # Openers that read as greetings rather than questions.
        "how are you",
        "how are you doing",
        "hows it going",
        "how is it going",
        "whats up",
        "sup",
        "are you there",
        "you there",
        "testing",
        "test",
        # Acknowledgements — the user is closing a turn, not opening one.
        "thanks",
        "thank you",
        "thanks a lot",
        "thanks so much",
        "thank you so much",
        "thank you very much",
        "many thanks",
        "much appreciated",
        "appreciate it",
        "cheers",
        "ty",
        "ok",
        "okay",
        "cool",
        "nice",
        "great",
        "perfect",
        "got it",
        "makes sense",
        "understood",
        # Farewells.
        "bye",
        "goodbye",
        "good bye",
        "see you",
        "see ya",
        "good night",
        "goodnight",
        # Meta: what this thing is, not what is in it.
        "help",
        "who are you",
        "what are you",
        "what can you do",
        "what do you do",
        "what is this",
        "what can i ask",
        "what can i ask you",
        "what should i ask",
        "how do you work",
        "how does this work",
        "how do i use this",
    }
)
"""Whole messages that are conversational rather than questions about the corpus.

Curated and closed on purpose. Adding a phrase here should pass one test: could
this ever be a question whose answer lives in someone's documents? If yes, it
does not belong — a false negative costs a chatty reply, a false positive costs
a citable answer.
"""


def normalise(message: str) -> str:
    """Reduce a message to bare lowercase words for matching.

    Strips case, punctuation, and apostrophes and collapses whitespace, so
    ``"Hi!!"``, ``"  hi  "`` and ``"Hi."`` are all the same message, and
    ``"how's it going?"`` matches ``"hows it going"``.
    """
    lowered = message.strip().lower()
    for apostrophe in _APOSTROPHES:
        lowered = lowered.replace(apostrophe, "")
    return _WHITESPACE.sub(" ", _PUNCTUATION.sub(" ", lowered)).strip()


def is_small_talk(message: str) -> bool:
    """Whether ``message`` is conversational rather than a question about the corpus.

    ``True`` only for a whole message in :data:`SMALL_TALK`. An empty message is
    not small talk — it has nothing to reply to and belongs on the normal path.
    """
    return normalise(message) in SMALL_TALK
