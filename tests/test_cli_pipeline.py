"""End-to-end CLI test: `ingest` then `query` as two separate invocations.

This is the issue's headline check — a developer ingests a folder and asks a
question with fakes and no network. Each ``main([...])`` call is an independent
run reading and writing only the shared state file, exactly as two shell
commands would, proving the corpus persists between them.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from ragsage.cli import main


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "france.txt").write_text("Paris is the capital of France. It sits on the Seine.")
    (docs / "biology.txt").write_text("The mitochondrion is the powerhouse of the cell.")
    return docs


def test_ingest_then_query_runs_the_full_loop(
    corpus: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    store = tmp_path / "state.json"

    ingest_code = main(["ingest", str(corpus), "--store", str(store)])
    ingest_out = capsys.readouterr().out
    assert ingest_code == 0
    assert "france.txt" in ingest_out
    assert store.exists()  # corpus persisted for the next process

    query_code = main(["query", "What is the capital of France?", "--store", str(store)])
    query_out = capsys.readouterr().out
    assert query_code == 0
    assert "Paris" in query_out
    assert "Sources:" in query_out  # a citation was rendered


def test_query_reports_honest_not_found(
    corpus: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    store = tmp_path / "state.json"
    main(["ingest", str(corpus), "--store", str(store)])
    capsys.readouterr()

    code = main(["query", "Who won the 1998 world cup final?", "--store", str(store)])
    out = capsys.readouterr().out

    assert code == 0
    assert "couldn't find" in out
    assert "Sources:" not in out


def test_reingest_is_reported_as_duplicate(
    corpus: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    store = tmp_path / "state.json"
    main(["ingest", str(corpus), "--store", str(store)])
    capsys.readouterr()

    main(["ingest", str(corpus), "--store", str(store)])
    out = capsys.readouterr().out

    assert "duplicate" in out


def test_query_without_a_corpus_errors_cleanly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["query", "anything", "--store", str(tmp_path / "missing.json")])
    out = capsys.readouterr().out

    assert code == 1
    assert "ingest" in out  # points the user at the fix
