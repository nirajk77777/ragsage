"""OpenAI-backed adapters for the generation-side model ports.

Generation, contextualization, query rewriting, and vision/OCR are hard-bound to
OpenAI (ADR-0002) and reach it directly through LangChain's :class:`ChatOpenAI` —
there is no gateway in the path. Each adapter conforms *by shape* to a port in
:mod:`ragsage.ports` (``LLMClient``, ``Contextualizer``, ``QueryRewriter``) and
holds one of the singleton clients built in :mod:`ragsage.providers.clients`; it
is a thin per-role wrapper, cheap enough to bind around the shared client for a
single request.

Conforming by shape, not by inheritance, is the point: these adapters are the
same kind of citizen as anything a consumer writes themselves, and the engine
still knows nothing about OpenAI. Swapping one out means passing a different
object to :class:`~ragsage.query.QueryEngine`, not editing the engine.

Each adapter also carries an optional
:data:`~ragsage.providers.config.CallConfig` and threads it into every
``ChatOpenAI`` call. It is opaque to ragsage: a consumer that wants per-user
usage attribution puts its own callbacks in there and we pass it through
untouched. It is bound per adapter instance rather than per method call because
the ports pass no per-call context — so a caller who needs different attribution
per request binds a new adapter (they are trivially cheap) around the shared
client. With nothing bound, the config is ``None`` and calls are simply
unobserved.
"""

from __future__ import annotations

import base64
from collections.abc import AsyncIterator, Sequence

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from ragsage.models import Chunk, Document, PageImage, Turn
from ragsage.providers.config import CallConfig

_CONTEXT_INSTRUCTION = (
    "Here is a chunk from the document above:\n<chunk>\n{chunk}\n</chunk>\n\n"
    "Give a short, standalone context (one or two sentences) that situates this "
    "chunk within the whole document to improve search retrieval. Answer with the "
    "context only, no preamble."
)

_REWRITE_INSTRUCTION = (
    "You rewrite a follow-up question so it stands alone. Given the conversation "
    "so far and a follow-up, resolve every pronoun and reference to what it points "
    "at and return one self-contained question that means the same thing without "
    "the conversation. If the follow-up is already standalone, return it unchanged. "
    "Reply with only the rewritten question, no preamble or quotes."
)

_TRANSCRIBE_INSTRUCTION = (
    "Transcribe all text in this page image exactly, preserving reading order. "
    "Return only the transcription."
)


def _text_of(message: BaseMessage) -> str:
    """Pull plain text out of a chat reply, whether it's a string or blocks.

    OpenAI chat replies are normally a bare string, but LangChain also models
    structured content as a list of blocks; join the text ones so a caller always
    gets a single string regardless of shape.
    """
    content = message.content
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


class OpenAIContextualizer:
    """``Contextualizer`` — situates a chunk with a cheap OpenAI model.

    The whole document is sent once as a system message and the per-chunk ask as
    the human message. That keeps the long, shared document body as the literal
    prompt *prefix* across every chunk of the same document, so OpenAI's automatic
    prefix caching reuses it — no Anthropic-style ``cache_control`` annotation is
    needed (or sent). The embed text is the model's context prepended to the
    verbatim chunk; if the model returns nothing usable, the raw chunk is embedded
    unchanged rather than polluting the vector with empty context.
    """

    def __init__(self, client: ChatOpenAI, *, config: CallConfig | None = None) -> None:
        self._client = client
        self._config = config

    async def contextualize(self, document: Document, chunk: Chunk, *, full_text: str) -> str:
        messages = [
            SystemMessage(content=f"<document>\n{full_text}\n</document>"),
            HumanMessage(content=_CONTEXT_INSTRUCTION.format(chunk=chunk.text)),
        ]
        reply = await self._client.ainvoke(messages, config=self._config)
        context = _text_of(reply).strip()
        if not context:
            return chunk.text
        return f"{context}\n\n{chunk.text}"


class OpenAIQueryRewriter:
    """``QueryRewriter`` — condenses a follow-up against history with a cheap model.

    The conversation is rendered as a short transcript and handed to the same
    cheap OpenAI tier contextualization uses; the model returns a standalone
    question that :class:`~ragsage.query.QueryEngine` retrieves and generates on.
    With no history there is nothing to resolve, so the question is returned
    unchanged and no call is made at all. If the model returns nothing usable, the
    original question is used rather than retrieving on empty text.
    """

    def __init__(self, client: ChatOpenAI, *, config: CallConfig | None = None) -> None:
        self._client = client
        self._config = config

    async def rewrite(self, question: str, history: Sequence[Turn]) -> str:
        if not history:
            return question
        transcript = "\n".join(f"User: {t.question}\nAssistant: {t.answer}" for t in history)
        content = (
            f"{_REWRITE_INSTRUCTION}\n\nConversation:\n{transcript}\n\n"
            f"Follow-up question: {question}\n\nStandalone question:"
        )
        reply = await self._client.ainvoke([HumanMessage(content=content)], config=self._config)
        return _text_of(reply).strip() or question


class OpenAILLMClient:
    """``LLMClient`` — streamed generation and multimodal vision transcription.

    Generation streams token-by-token from the generation client so a caller can
    relay each delta straight to its own transport; transcription sends the page
    image to the vision client as a multimodal message and returns the whole reply
    at once. Two clients rather than one because they differ in streaming and in
    token budget.
    """

    def __init__(
        self,
        *,
        generation: ChatOpenAI,
        vision: ChatOpenAI,
        config: CallConfig | None = None,
    ) -> None:
        self._generation = generation
        self._vision = vision
        self._config = config

    async def generate(self, prompt: str) -> AsyncIterator[str]:
        async for chunk in self._generation.astream(
            [HumanMessage(content=prompt)], config=self._config
        ):
            token = _text_of(chunk)
            if token:
                yield token

    async def transcribe(self, image: PageImage) -> str:
        if image.data is None:
            # No bytes in hand means the parser couldn't render the page; there is
            # nothing to send a vision model. The caller treats this as empty text.
            return ""
        data_uri = "data:image/png;base64," + base64.b64encode(image.data).decode()
        message = HumanMessage(
            content=[
                {"type": "text", "text": _TRANSCRIBE_INSTRUCTION},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ]
        )
        reply = await self._vision.ainvoke([message], config=self._config)
        return _text_of(reply)
