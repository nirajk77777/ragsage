# Failure modes of the heuristic parser

When an answer looks wrong, check the parser before you check retrieval: often the engine
retrieved exactly what it was given, and what it was given is not what you saw on the page.
`ragsage` parses with a pure-Python, model-free heuristic
parser: no layout model, no table model, no OCR tier. That is a deliberate trade, and the
degradation on messy inputs is accepted rather than unforeseen: what it buys is a parser
that installs anywhere, pulls no heavyweight numerical stack, and downloads no model
weights. This document is what "accepted degradation" actually looks like from the asking
side, so you can recognise it in one reading instead of spending an afternoon on the
retrieval stack.

Each entry is **symptom → how to confirm → why → workaround**, and names the code that
decides. Line numbers drift; the function or constant beside each one is the durable part.

Where a workaround does not exist, it says so and says what building one would take.

1. [A two-column page comes back interleaved — or as a table](#1-a-two-column-page-comes-back-interleaved--or-as-a-table)
2. [A borderless table loses its columns, or tears the page apart](#2-a-borderless-table-loses-its-columns-or-tears-the-page-apart)
3. [A text page is read by the vision model, and its real text is discarded](#3-a-text-page-is-read-by-the-vision-model-and-its-real-text-is-discarded)
4. [A figure the vision route should have rescued stays on the text route](#4-a-figure-the-vision-route-should-have-rescued-stays-on-the-text-route)
5. [Headings vanish on uniformly typeset documents — and contextualisation with them](#5-headings-vanish-on-uniformly-typeset-documents--and-contextualisation-with-them)
6. [Unsupported by design: scans without a vision adapter, legacy Office, spreadsheets, bare images](#6-unsupported-by-design)
7. [A wrong declared media type silently wins](#7-a-wrong-declared-media-type-silently-wins)

## First: look at what the parser saw

Every entry below is diagnosed from the same twenty seconds of output — the Markdown the
parser produced per page, the layout signals it measured, and the metadata the chunker
attached:

```python
from ragsage.models import RawSource
from ragsage.parsing import HeuristicBackend

backend = HeuristicBackend()
parsed = backend.parse(RawSource(name="report.pdf", path="report.pdf"))

for page in parsed.pages:
    print(f"--- page {page.number}  {page.layout}  image={page.image is not None}")
    print(page.text)

for chunk in backend.chunk(parsed.document, parsed.pages, size=512, overlap=64):
    print(chunk.ordinal, dict(chunk.metadata), repr(chunk.text[:80]))
```

Read it as three questions. *Is the prose in the right order?* *Are the tables tables?* *Does
each chunk carry a `headings` path?* If all three are yes and answers are still wrong, the
problem is downstream of this document.

Two notes on the snippet. `parse` stashes its pages for the matching `chunk` call and that
stash is consumed once (`HeuristicBackend._take_parsed`,
[`backend.py:103`](https://github.com/nirajk77777/ragsage/blob/main/src/ragsage/parsing/backend.py), popped at `backend.py:126`), so call
`chunk` once per `parse`.
And this is the parser alone — no embedder, no store, no network.

## 1. A two-column page comes back interleaved — or as a table

**Symptom.** The answer quotes a sentence that does not exist in the document: it reads like
two half-sentences spliced together, one from each column. Or a page of ordinary two-column
prose is quoted as a *table*, pairing an unrelated left-hand and right-hand line in the same
"row".

**How to confirm.** Dump the page Markdown. Interleaving looks like alternating left/right
content on consecutive lines. The table version is unmistakable — a `| … | … |` block whose
two columns are each a vertical slice of the page. Compare against the plain case: a
two-column page that parsed *correctly* emits the whole left column first, then the whole
right column.

**Why it happens.** Reading order comes from `_column_bands`
([`pdf.py:304`](https://github.com/nirajk77777/ragsage/blob/main/src/ragsage/parsing/pdf.py)), which finds columns by locating the vertical
gutter between word clusters. Two things defeat it:

- **Any full-width element whose words cross the gutter** — a centred title, a wide figure
  caption, a running header — fills the gap. Word x-spans are accumulated over *every* word
  on the page and merged (`pdf.py:320-326`), so once ink sits in the gutter there is no gap
  left that clears the `max(18.0, page_width * 0.04)` threshold (`pdf.py:328`) — 24.5pt on
  US Letter. The page collapses to a single band, and `_group_lines`
  ([`pdf.py:259`](https://github.com/nirajk77777/ragsage/blob/main/src/ragsage/parsing/pdf.py)) then merges left- and right-column lines
  that share a baseline into one line, within a tolerance of `max(3.0, height * 0.6)`
  (`pdf.py:270`).
- **A column that stops early** — the last page of an article, a short sidebar — fails
  `_both_sides_span_height` ([`pdf.py:342`](https://github.com/nirajk77777/ragsage/blob/main/src/ragsage/parsing/pdf.py)), which requires
  text on *both* sides of the candidate gutter to span at least
  `_MIN_COLUMN_HEIGHT_FRAC = 0.6` of the page's text height (`pdf.py:301`). Same collapse.

The table version is the second-order effect: those merged lines now contain a wide
inter-word gap where the gutter used to be, which is exactly what the borderless-table
detector looks for (entry 2), so a run of them is claimed as a two-column table.

**Workaround.** None by configuration — the gutter fraction, the line tolerance and
`_MIN_COLUMN_HEIGHT_FRAC` are module constants, not settings. What works:

- **Supply a flow format instead.** DOCX, PPTX and HTML have no column geometry to misread;
  those paths walk native structure ([`docx.py`](https://github.com/nirajk77777/ragsage/blob/main/src/ragsage/parsing/docx.py),
  [`html.py`](https://github.com/nirajk77777/ragsage/blob/main/src/ragsage/parsing/html.py)) and are unaffected by this entry entirely.
- **Re-export the PDF single-column** if it is yours to re-export.
- **Bring your own parser.** `DocumentParser` is a `Protocol`
  ([`ports.py:48`](https://github.com/nirajk77777/ragsage/blob/main/src/ragsage/ports.py)); see
  [`examples/custom_parser.py`](https://github.com/nirajk77777/ragsage/blob/main/examples/custom_parser.py). You can parse the hard
  documents your way and keep the rest of the engine.

What a fix needs: exclude full-width lines from the span accumulation before computing
gutters (so a title cannot mask them), and replace the both-sides-height test with a per-band
ink-density test (so a short column still counts as a column). Neither is large; neither is
done.

## 2. A borderless table loses its columns, or tears the page apart

**Symptom.** Four shapes, all of which read to a user as *the answer cites the right page and
quotes the wrong cell*:

1. Two columns arrive as one. A three-column table comes back as two, with `Region Q1` as a
   single header cell — so a question about Q1 gets an answer holding a label and a number
   glued together.
2. No table at all: the rows come back as prose, values fused to their labels (`RegionQ1 Q2`).
3. The table's columns are separated into different blocks entirely — every label first, then
   every value — so "North" and "10" no longer sit near each other and the answer pairs the
   wrong ones.
4. The table is quoted with rows silently missing: it stops at the first row that had a blank
   cell, and everything below that row is simply not in the table any more.

**How to confirm.** Dump the page Markdown and find the table. A pipe table with fewer
columns than the PDF is shape 1; no `| … | … |` block at all is shape 2; several paragraph
blocks each holding one column of the table is shape 3; a pipe table with fewer rows than the
PDF is shape 4. Compare cell for cell against the PDF — it is worth the minute.

**Why it happens.** Tables drawn with rules are found by pdfplumber's line strategy and
trusted outright (`_RULED_SETTINGS` [`pdf.py:66`](https://github.com/nirajk77777/ragsage/blob/main/src/ragsage/parsing/pdf.py),
`_ruled_tables` `pdf.py:415`). Everything else goes through `_borderless_tables`
([`pdf.py:430`](https://github.com/nirajk77777/ragsage/blob/main/src/ragsage/parsing/pdf.py)), which is inference over whitespace:

- **The cell boundary is one fixed distance, applied per line.** A line splits into cells
  wherever an inter-word gap reaches `_COLUMN_GAP = 24.0` points (`pdf.py:74`, applied in
  `_split_cells` [`pdf.py:470`](https://github.com/nirajk77777/ragsage/blob/main/src/ragsage/parsing/pdf.py)) — nothing aligns the splits
  between one row and the next. So the split depends on how *wide the text in each cell
  happens to be*, not on where the column is. Verified on the same three-column table at
  three column pitches: at 60pt every line splits into 3 cells (correct); at 45pt every line
  splits into 2 (`['Region Q1', 'Q2']` — shape 1, a whole column absorbed into its
  neighbour); at 36pt the lines split into 1, 1 and 2 cells respectively, which fails the run
  test below and yields no table at all (shape 2).
- **A widely set table can be torn up before the detector even runs.** Its inter-column gaps
  clear the 24.5pt gutter threshold, and if the table is most of the page's content its gaps
  trivially span the page's text height, so `_column_bands` (entry 1) treats them as page
  columns and `_serialize` ([`pdf.py:179`](https://github.com/nirajk77777/ragsage/blob/main/src/ragsage/parsing/pdf.py)) emits each band
  separately — shape 3. The comment at `pdf.py:297-300` explains why this is *supposed* to be
  safe; the both-sides-height test is exactly what stops protecting you when the table is the
  page.
- **A blank cell ends the table.** A table is only a table where consecutive lines split into
  the *same* number of cells (`pdf.py:450`) for at least `_MIN_TABLE_ROWS = 2` lines
  (`pdf.py:77`). A row with an empty middle cell splits into fewer cells and ends the run;
  the rows below it only form a second table if *they* agree with each other. Verified on a
  four-row table whose third row has a blank middle cell: the emitted table is the header plus
  **one** row, and the last two rows leave the table entirely. Note too that the first line of
  whatever run survives becomes the header (`_table_to_markdown`
  [`pdf.py:484`](https://github.com/nirajk77777/ragsage/blob/main/src/ragsage/parsing/pdf.py)) — so where a run starts below the real
  header, a data row is promoted into its place and every value is read under the wrong column
  name (shape 4).

**Workaround.**

- **Rule your tables.** If you control the source, a table with drawn rules takes the trusted
  path and none of this applies.
- **Supply DOCX, PPTX or HTML.** Those paths read real table grids
  (`_render_table` [`docx.py:185`](https://github.com/nirajk77777/ragsage/blob/main/src/ragsage/parsing/docx.py),
  [`html.py:197`](https://github.com/nirajk77777/ragsage/blob/main/src/ragsage/parsing/html.py)) rather than guessing from whitespace.
- **Or take the table out of the document** — a CSV or Markdown table ingested as text goes
  through the text path untouched.
- There is no setting to tune. `_COLUMN_GAP` and `_MIN_TABLE_ROWS` are module constants.

What a fix needs: cell boundaries inferred from x-position *clustering across the whole
candidate block* rather than a fixed per-line gap, so column edges are shared between rows
and a row with a blank cell keeps its shape. That is a real piece of work — it is the part of
a layout model we chose not to ship.

## 3. A text page is read by the vision model, and its real text is discarded

**Symptom.** Quotes from some pages do not match the document verbatim — a citation points at
text that is *close to* but not identical to what is printed, because a model re-read the page
instead of the parser reading its text layer. Ingest is also slower and more expensive than the
document's size suggests. The affected pages tend to be the ones with a chart or a photo on
them.

**How to confirm.** Two ways. In the dump, look at the printed `layout` and `image=` for the
suspect page. And at pipeline level, `IngestionResult.route_counts`
([`ingestion.py:57`](https://github.com/nirajk77777/ragsage/blob/main/src/ragsage/ingestion.py)) reports how many pages took each route —
the CLI surfaces the same thing as `, N via vision` (`cli.py:159-160`), and the tracer emits
a `page_routed` event per page (`ingestion.py:153`). A text-heavy document with a non-zero
vision count is this entry.

**Why it happens.** `LayoutPageClassifier._has_usable_text_layer`
([`classifier.py:66`](https://github.com/nirajk77777/ragsage/blob/main/src/ragsage/parsing/classifier.py)) disqualifies a page's text layer
if *any one* of three thresholds trips (`classifier.py:71`, `:73`, `:75`). The third is
`min_bitmap_area = 50_000.0` (`classifier.py:51`) — the area of the single largest embedded
bitmap, in the page's own units, and **it does not consult `text_chars` at all**. At 72 points
to the inch, 50,000 pt² is about 3.5 × 2.8 inches. So a perfectly readable page with 1,200
characters of text and one 240 × 210pt figure (50,400 pt²) routes to vision, while the same
page with a 200 × 150pt figure (30,000 pt²) does not.

That is not merely a wasted call: the pipeline *replaces* the page's extracted text with the
transcription (`replace(page, text=text)`, [`ingestion.py:149-150`](https://github.com/nirajk77777/ragsage/blob/main/src/ragsage/ingestion.py)),
so good born-digital text is thrown away in favour of a model's reading of a 144-DPI render
(`_RENDER_SCALE`, [`pdf.py:56`](https://github.com/nirajk77777/ragsage/blob/main/src/ragsage/parsing/pdf.py)).

**Workaround.** This one *is* configurable. Build the classifier yourself and inject it:

```python
from ragsage.parsing import LayoutPageClassifier

# Only page-filling rasters should force the vision route, not ordinary figures.
classifier = LayoutPageClassifier(min_bitmap_area=250_000.0)
```

`IngestionPipeline` takes it as its `classifier` argument
([`ingestion.py:68`](https://github.com/nirajk77777/ragsage/blob/main/src/ragsage/ingestion.py)). Raising `min_bitmap_area` (or
`max_image_area_ratio`, `classifier.py:50`) works as expected: the parser may still have
attached a rendered image to the page, and it simply goes unused. Lowering the thresholds is
the direction that does **not** work — see entry 4.

What a better default needs: the bitmap-area test should be conditional on the text layer
being thin, not independent of it. Changing that default is a behaviour change to the
routing policy, so it wants its own ticket and its own measurement rather than a quiet edit.

## 4. A figure the vision route should have rescued stays on the text route

**Symptom.** A chart, diagram, schematic or map in your document is un-askable. Questions
about it return "not found", or an answer built only from its caption. Nothing in the ingest
output suggests anything went wrong — no error, no vision call, a normal chunk count.

**How to confirm.** In the dump, the page's Markdown is a caption and little else, its
`layout` reads `image_area_ratio=0.0, bitmap_area=0.0`, and `image=False`. `route_counts`
shows zero vision pages.

**Why it happens.** Two mechanisms compound.

- `_measure_layout` ([`pdf.py:148`](https://github.com/nirajk77777/ragsage/blob/main/src/ragsage/parsing/pdf.py)) measures raster images
  only — it reads `page.images` (`pdf.py:159`). A figure drawn in **vector** graphics (most
  charts exported from plotting tools, CAD drawings, anything built of paths and rectangles)
  contributes no image area and no bitmap area whatsoever, so the page never looks
  image-dominated. Verified: a page holding a vector bar chart plus a 46-character caption
  measures `text_chars=46, image_area_ratio=0.0, bitmap_area=0.0` and routes to `TEXT`.
- Tuning the classifier cannot rescue it. Whether a page gets a rendered image is decided
  *inside the parser*, by a module-level classifier built with default thresholds
  (`_CLASSIFIER = LayoutPageClassifier()`, [`pdf.py:48`](https://github.com/nirajk77777/ragsage/blob/main/src/ragsage/parsing/pdf.py), used
  by `_routes_to_vision` `pdf.py:126` and the attachment at `pdf.py:119-122`). And
  `classify` returns `TEXT` whenever there is no image to transcribe
  ([`classifier.py:62-64`](https://github.com/nirajk77777/ragsage/blob/main/src/ragsage/parsing/classifier.py)) — sensibly, since there
  would be nothing to send. So a classifier you inject with *lower* thresholds still routes
  that page to `TEXT`: the image it would have needed was never rendered. Verified with
  `LayoutPageClassifier(min_text_chars=500, max_image_area_ratio=0.01, min_bitmap_area=1.0)`
  — still `TEXT`.

The rescue that *does* work is the degrade path: a page pdfplumber throws on is rendered and
forced to vision regardless of measurement (`_parse_page`'s `except`,
[`pdf.py:107-117`](https://github.com/nirajk77777/ragsage/blob/main/src/ragsage/parsing/pdf.py), with `_VISION_FALLBACK_LAYOUT`
`pdf.py:52`). A page that parses *successfully* but yields almost nothing useful gets no such
treatment.

**Workaround.** There isn't one inside the library today, and that is the honest answer. What
helps at the edges: export the figure page as an image so it arrives as a raster (which the
measurement does see), or supply the figure's underlying numbers as CSV/Markdown alongside
the document. Trimming the caption below `min_text_chars = 24` (`classifier.py:49`) does
technically flip the route, but tuning your document to a threshold is not advice.

What a fix needs: a vector-ink signal in `_measure_layout` — coverage computed from
`page.curves` / `page.rects` / `page.lines` — carried as a fourth field on `PageLayout` and a
fourth threshold on the classifier. Then a vector-heavy page with a thin text layer would
route like the scan it effectively is. Until that exists, vector figures are outside what
this parser can see.

One related detail worth knowing: `_render_png` returns `None` if rendering fails
(`pdf.py:522`, and it is deliberately not covered by tests). The page still carries a
`PageImage` and still routes to vision, so the transcription then depends entirely on your
`LLMClient.transcribe` ([`ports.py:126`](https://github.com/nirajk77777/ragsage/blob/main/src/ragsage/ports.py)) being able to resolve the
image's `ref` on its own.

## 5. Headings vanish on uniformly typeset documents — and contextualisation with them

**Symptom.** Answers cite the right page but the quoted passage starts mid-thought, or spans
two sections that have nothing to do with each other. Citations show no section — the UI has
a page number and nothing else to say about where the passage sits. Retrieval feels blunt on
a document that is, visually, clearly organised.

**How to confirm.** Look at the chunk metadata in the dump. `{}` on every chunk of a page
means no heading was detected there; a healthy page shows
`{'headings': ['1 Introduction', ...]}`. In the page Markdown, no `#` lines at all is the same
finding seen upstream.

**Why it happens.** Heading detection is a font-size argument. `_modal_size`
([`pdf.py:358`](https://github.com/nirajk77777/ragsage/blob/main/src/ragsage/parsing/pdf.py)) takes the most common rounded font size on the
page as the body size, and `_is_heading` ([`pdf.py:373`](https://github.com/nirajk77777/ragsage/blob/main/src/ragsage/parsing/pdf.py))
promotes a line only if it is **at least one point larger than body and at most
`_MAX_HEADING_WORDS = 14` words** (`pdf.py:385`, `pdf.py:59`) — or, at body size, if it is
short, carries a leading section number (`_NUMBER_PREFIX`, `pdf.py:62`) *and* is bold
(`pdf.py:389`).

A document that marks sections by weight or spacing alone — the common house style for
reports and legal documents — trips none of that. Verified against a page whose section
titles are set in the same 11pt face as the body: zero headings detected. Testing the
same-size rule directly, only **numbered *and* bold** is promoted; numbering alone is not,
and bold alone is not. A heading longer than 14 words is never a heading regardless.

The damage is downstream, in two places:

- **Chunk boundaries.** Pass 1 of the chunker splits on ATX headings
  (`_sections` [`chunking.py:118`](https://github.com/nirajk77777/ragsage/blob/main/src/ragsage/parsing/chunking.py), levels at
  `chunking.py:38`). With no headings the whole page is one section (`chunking.py:132`), so
  the only boundaries left are the token-budget cuts made by `_split_prose`
  (`chunking.py:225`) — arbitrary with respect to meaning.
- **Contextualisation.** `HeadingWindowContextualizer` builds its context header from
  `metadata["headings"]` (`_heading_path`
  [`contextualizing.py:239`](https://github.com/nirajk77777/ragsage/blob/main/src/ragsage/contextualizing.py), read at
  `contextualizing.py:246`) and emits the `Section:` line only when that path is non-empty
  (`_header`, `contextualizing.py:158-164`). On a flat document the header degrades to
  `Document: <name>` alone. Verified: the contextualised `embed_text` for such a page begins
  `Document: uniform.pdf` with no `Section:` line at all. So the same documents that chunk
  badly also embed with less context — the two weaknesses stack rather than offsetting.

**Workaround.**

- **Supply DOCX or HTML where you can.** Those paths read declared structure, not typography:
  `Heading N` / `Title` styles (`_heading_level` [`docx.py:125`](https://github.com/nirajk77777/ragsage/blob/main/src/ragsage/parsing/docx.py))
  and `<h1>`…`<h6>` (`_HEADINGS` [`html.py:59`](https://github.com/nirajk77777/ragsage/blob/main/src/ragsage/parsing/html.py)). A document
  that is flat in PDF is often perfectly structured in its source format.
- **Number your sections, in bold**, if you are generating the PDF. `1 Introduction` in bold
  at body size is detected and nests correctly — `_render_heading` (`pdf.py:394`) deepens the
  level from the number's dot count, so `1.1 Background` lands under `1 Introduction`.
- The `HeadingWindowContextualizer`'s neighbour window still works without headings — it
  locates the chunk in the page text and quotes the surrounding paragraphs
  (`_neighbours`, `contextualizing.py:166`). It is a partial substitute for the missing
  section line, not a replacement.

## 6. Unsupported by design

Four things this parser does not do. If you are here about one of them, stop debugging.

**Scanned pages have no local OCR tier.** The model-free parser rules one out by
construction — an OCR tier is a model, and shipping one would cost the portability the
parser exists to keep. A scan is readable only if the caller wired an `LLMClient` whose
`transcribe` ([`ports.py:126`](https://github.com/nirajk77777/ragsage/blob/main/src/ragsage/ports.py)) actually calls a vision model. Routing works — a
page-filling raster with no text layer trips all three classifier thresholds and gets a
rendered image — but the transcription is your adapter's job, not the library's. Driving the
engine from the CLI or the in-memory fakes gives you a stand-in instead:
`FakeLLMClient.transcribe` ([`fakes.py:307`](https://github.com/nirajk77777/ragsage/blob/main/src/ragsage/fakes.py)) decodes the image bytes
as UTF-8, or returns `[vision transcription of <ref>]` when there are none. If a scanned
document ingested with implausibly few or implausibly odd chunks, check what your `transcribe`
returns.

**Legacy binary Office formats — `.doc`, `.xls`, `.ppt` — are rejected, deliberately.** They
appear in none of the router's three tables (`_MEDIA_TYPES`
[`router.py:42`](https://github.com/nirajk77777/ragsage/blob/main/src/ragsage/parsing/router.py), `_EXTENSIONS` `router.py:58`,
`_SNIFF_EXTENSIONS` `router.py:73`), so an OLE2 file resolves to `UNKNOWN` and
`HeuristicBackend._parse_pages` raises
`ValueError: unrecognised document format for source '…'`
([`backend.py:99-100`](https://github.com/nirajk77777/ragsage/blob/main/src/ragsage/parsing/backend.py)) — verified for `.doc`, `.xls` and
`.ppt`. This is the intended behaviour: the legacy binary formats are out of scope for the
heuristic parser and for the current spec alike. Convert to the modern OOXML equivalent
(`.docx` / `.pptx`) and re-upload.

**Spreadsheets are out of scope — but they do not fail cleanly, and that is a wart.** `.xlsx`
has no entry in the media-type or extension tables either, so routing falls through to
content sniffing, where `puremagic` reports every OOXML zip as the same ambiguous match list
(`.docx`, `.pptx`, `.xlsx`, …) and `_from_content` takes the first entry that maps
(`router.py:134-137`) — which is `.docx`. So a workbook is routed to the DOCX path and then
fails somewhere inside `python-docx`, rather than with the clear "unrecognised document format"
message above. Verified against an OOXML zip: `route()` returns `DocumentFormat.DOCX`, both
with no declared media type and with the correct spreadsheet one. Export the sheet as CSV and
ingest it as text.

**A bare image file is not a document.** `.png`, `.jpg` and `.tif` uploads resolve to
`UNKNOWN` (the sniff table maps no image formats, and the bytes do not decode as text —
`_looks_like_text` `router.py:141`) and are rejected with the same `ValueError`. Verified for
all three. If the image is a scanned page, wrap it in a PDF; then the vision route applies.

## 7. A wrong declared media type silently wins

**Symptom.** A document reports a successful ingest, but everything retrieved from it is
gibberish — binary noise, `%PDF-1.3` and object dictionaries appearing in answers, or one
enormous chunk of junk.

**How to confirm.** Ask the router what it decided:

```python
from ragsage.models import RawSource
from ragsage.parsing import route

source = RawSource(name="report.pdf", path="report.pdf", media_type="text/plain")
print(route(source, open("report.pdf", "rb").read()))  # -> DocumentFormat.TEXT
```

**Why it happens.** `route` ([`router.py:82`](https://github.com/nirajk77777/ragsage/blob/main/src/ragsage/parsing/router.py)) trusts the
caller's `media_type` first and returns immediately on a hit (`router.py:91-93`), ahead of the
extension and the content sniff. A PDF declared `text/plain` therefore takes the text path,
where the bytes are decoded with `errors="replace"`
(`backend.py:97`) — so it "succeeds", producing replacement characters and PDF syntax as
document text. Verified: the first page's text begins `%PDF-1.3`. The same trap catches an
RTF, which is not a format this parser supports but *is* valid text, so it ingests as raw
`{\rtf1\ansi…}` markup rather than being rejected.

**Workaround.** Pass a correct `media_type`, or pass `None`
([`models.py:53`](https://github.com/nirajk77777/ragsage/blob/main/src/ragsage/models.py)) and let the extension and the content sniff
decide — they are more reliable than an upstream `Content-Type` header in practice. If you
are ingesting from an HTTP upload, be sceptical of what the browser declared.

## Not on this list

These are working as intended and are not worth investigating:

- **A table with drawn rules coming through intact.** Rules are strong evidence, so those
  tables take pdfplumber's line strategy and skip the whitespace inference entirely
  (`_ruled_tables`, [`pdf.py:415`](https://github.com/nirajk77777/ragsage/blob/main/src/ragsage/parsing/pdf.py)). Entry 2 is about the
  borderless detector, not this path.
- **A page that failed to parse still producing a chunk.** By design: it degrades to the
  vision route rather than failing the document (`pdf.py:107-117`).
- **Flow formats all landing on page 1.** DOCX, PPTX, HTML and text have no page grid;
  `_FLOW_FORMAT_PAGE` ([`backend.py:43`](https://github.com/nirajk77777/ragsage/blob/main/src/ragsage/parsing/backend.py)) collapses them
  onto one page number deliberately.
- **`.md` and `.csv` ingesting as plain text.** That is the text path doing its job.

## Reporting one of these

If you hit something not described here, the useful bug report is the dump from
[First: look at what the parser saw](#first-look-at-what-the-parser-saw) — the page Markdown,
the `PageLayout`, and the chunk metadata — plus a redacted page that reproduces it. The parser
is pure Python and its tests build PDFs in-process with `reportlab`
([`tests/test_pdf_path.py`](https://github.com/nirajk77777/ragsage/blob/main/tests/test_pdf_path.py)), so a reproduction can usually be
turned straight into a failing test.
