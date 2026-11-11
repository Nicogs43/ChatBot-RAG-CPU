import pytest
from conftest import FakeModel
from langchain_core.documents import Document

import main_gradio as ui


def _pending_history():
    return [{"role": "user", "content": "q"}, {"role": "assistant", "content": ui.WAITING_NOTE}]


def test_sources_show_page_as_pdf_page_number_and_relevance(docs):
    docs[0].metadata["relevance_score"] = 0.938
    page_html = ui.render_sources(docs)
    assert "Page 3" in page_html and "Page 8" in page_html  # metadata pages 2 and 7 are zero-based
    assert "relevance 0.94" in page_html
    assert page_html.count('class="mc-passage"') == 2


def test_passage_text_is_escaped():
    page_html = ui.render_sources([Document(page_content="<script>alert(1)</script> & more", metadata={"page": 0})])
    assert "<script>" not in page_html
    assert "&lt;script&gt;" in page_html and "&amp; more" in page_html


def test_no_passages_warns_that_the_answer_is_unsupported():
    assert "mc-warning" in ui.render_sources([])


def test_passage_without_page_metadata_still_renders():
    assert "Page unknown" in ui.render_sources([Document(page_content="text", metadata={})])


def test_speed_panel_lists_each_timing():
    page_html = ui.render_speed([("Search", "0.9 s"), ("Writing speed", "7.4 tokens/s")])
    assert page_html.count("<dt>") == 2 and "7.4 tokens/s" in page_html


@pytest.mark.parametrize(
    ("seconds", "text"), [(0.94, "0.9 s"), (59.94, "59.9 s"), (87.4, "1 min 27 s"), (425, "7 min 5 s")]
)
def test_durations_read_in_seconds_then_minutes(seconds, text):
    assert ui._seconds(seconds) == text


def test_message_text_accepts_string_and_gradio_parts():
    assert ui._message_text({"content": "hello"}) == "hello"
    parts = [{"type": "text", "text": "hel"}, {"type": "text", "text": "lo"}]
    assert ui._message_text({"content": parts}) == "hello"


def test_empty_question_is_refused():
    with pytest.raises(ui.gr.Error):
        ui.start_question("   ", [])


def test_question_is_shown_at_once_with_a_waiting_note():
    box, history, question, *_ = ui.start_question(" What winds? ", [])
    assert box == "" and question == "What winds?"
    assert history == [
        {"role": "user", "content": "What winds?"},
        {"role": "assistant", "content": ui.WAITING_NOTE},
    ]


def test_stop_marks_a_partial_answer_and_keeps_its_text():
    history = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": [{"type": "text", "text": "Channel"}]},
    ]
    history, *_ = ui.stop_answer(history)
    assert history[-1]["content"] == "Channel *(stopped)*"


def test_stop_before_the_first_word_replaces_the_waiting_note():
    history, *_ = ui.stop_answer(_pending_history())
    assert "Stopped before the first word" in history[-1]["content"]


def test_answer_streams_into_the_last_message_then_reports_timings(make_engine, monkeypatch):
    engine, _, _ = make_engine()
    monkeypatch.setattr(ui, "_engine", engine)
    updates = list(ui.answer_question("q", _pending_history()))
    assert "mc-passage" in updates[0][1]  # passages appear before the first word
    assert updates[-1][0][-1]["content"] == "Channel 12 and 16."
    for label in ("Search", "First word after", "Whole answer", "Writing speed"):
        assert label in updates[-1][2]


def test_failed_answer_is_reported_in_the_chat(make_engine, monkeypatch):
    engine, _, _ = make_engine(model=FakeModel(error=RuntimeError("out of memory")))
    monkeypatch.setattr(ui, "_engine", engine)
    updates = list(ui.answer_question("q", _pending_history()))
    assert "could not be generated: out of memory" in updates[-1][0][-1]["content"]


def test_page_builds_with_the_available_models():
    demo = ui.build_demo(["qwen_0.5b", "phi_int4"], "qwen_0.5b")
    assert isinstance(demo, ui.gr.Blocks)
