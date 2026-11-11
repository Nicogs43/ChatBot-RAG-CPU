import argparse
import gc
import html
import threading
import time

import gradio as gr

from chatbot import RagEngine, load_retrieval_resources
from config import available_models, default_model, model_label

EXAMPLES = [
    "What are the typical winds in Genoa?",
    "What is the VHF channel of the port of Imperia?",
    "Can you tell me the email address of the port of Ischia?",
    "What is the depth of the sea in the port of Alassio?",
]

WAITING_NOTE = "*Reading the passages. On a small CPU the first words can take a minute or more.*"

# ---------------------------------------------------------------------------
# Engine state: one model in memory at a time, shared by everyone using the page
# ---------------------------------------------------------------------------
_resources = None  # (vectorstore, reranker), loaded once
_engine = None
_engine_lock = threading.Lock()


def load_model(model_key: str) -> RagEngine:
    """Make `model_key` the model in memory, releasing the previous one first."""
    global _resources, _engine
    with _engine_lock:
        if _engine is not None and _engine.model_key == model_key:
            return _engine
        if _resources is None:
            _resources = load_retrieval_resources()
        _engine = None  # free the RAM of the old model before loading the new one
        gc.collect()
        _engine = RagEngine(model_key, *_resources)
        return _engine


# ---------------------------------------------------------------------------
# Rendering of the side panel
# ---------------------------------------------------------------------------
def render_sources(docs=None, note=None) -> str:
    if note:
        return f'<p class="mc-note">{html.escape(note)}</p>'
    if docs is None:
        return '<p class="mc-note">The passages behind each answer appear here, with their page in the guide.</p>'
    if not docs:
        return ('<p class="mc-note mc-warning">No passage in the guide matched this question closely enough. '
                "The answer comes from the model alone, so check it before relying on it.</p>")
    items = []
    for doc in docs:
        page = doc.metadata.get("page")
        page_label = f"Page {page + 1}" if isinstance(page, int) else "Page unknown"
        score = doc.metadata.get("relevance_score")
        score_label = f'<span class="mc-score">relevance {score:.2f}</span>' if score is not None else ""
        items.append(
            f'<li class="mc-passage"><div class="mc-passage-head"><span class="mc-page">{page_label}</span>'
            f"{score_label}</div><p>{html.escape(doc.page_content)}</p></li>"
        )
    return f'<ol class="mc-passages">{"".join(items)}</ol>'


def render_speed(rows=None, note=None) -> str:
    if note:
        return f'<p class="mc-note">{html.escape(note)}</p>'
    if not rows:
        return '<p class="mc-note">Timings for the latest answer appear here.</p>'
    cells = "".join(f"<div><dt>{html.escape(label)}</dt><dd>{html.escape(value)}</dd></div>" for label, value in rows)
    return f'<dl class="mc-speed">{cells}</dl>'


def _seconds(value: float) -> str:
    return f"{value:.1f} s" if value < 60 else f"{int(value // 60)} min {int(value % 60)} s"


# ---------------------------------------------------------------------------
# Event handlers
# ---------------------------------------------------------------------------
def start_question(question, history):
    """Show the question at once and switch the Ask button to Stop."""
    question = (question or "").strip()
    if not question:
        raise gr.Error("Type a question first.", print_exception=False)
    history = history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": WAITING_NOTE},
    ]
    return ("", history, question, render_sources(note="Searching the guide…"), render_speed(note="Measuring…"),
            gr.update(visible=False), gr.update(visible=True), gr.update(interactive=False))


def answer_question(question, history):
    """Retrieve, then stream the answer into the last chat message."""
    engine = _engine
    started = time.perf_counter()
    try:
        docs = engine.retrieve(question)
        search_time = time.perf_counter() - started
        yield history, render_sources(docs), gr.skip()

        answer, first_word_time = "", None
        for piece in engine.stream(question, docs):
            if first_word_time is None:
                first_word_time = time.perf_counter() - started
            answer += piece
            history[-1]["content"] = answer
            yield history, gr.skip(), gr.skip()
        total_time = time.perf_counter() - started

        history[-1]["content"] = answer.strip() or "The model returned an empty answer. Try rephrasing the question."
        rows = [("Search", _seconds(search_time))]
        if first_word_time is not None:
            n_tokens = len(engine.tokenizer(answer, add_special_tokens=False)["input_ids"])
            writing_time = max(total_time - first_word_time, 1e-6)
            rows += [
                ("First word after", _seconds(first_word_time)),
                ("Whole answer", _seconds(total_time)),
                ("Writing speed", f"{n_tokens / writing_time:.1f} tokens/s"),
            ]
        yield history, gr.skip(), render_speed(rows)
    except Exception as e:
        history[-1]["content"] = f"The answer could not be generated: {e}"
        yield history, gr.skip(), render_speed(note="No timings: the answer failed.")


def finish_question():
    return gr.update(visible=True), gr.update(visible=False), gr.update(interactive=True)


def _message_text(message) -> str:
    """Text of a chat message: Gradio hands content back as a string or as a list of parts."""
    content = message["content"]
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content if isinstance(part, dict))


def stop_answer(history):
    """Runs when Stop is pressed; Gradio has already closed the answer stream, which stops the model."""
    if history and history[-1]["role"] == "assistant":
        text = _message_text(history[-1])
        if text == WAITING_NOTE:
            history[-1]["content"] = "*Stopped before the first word.*"
        else:
            history[-1]["content"] = f"{text} *(stopped)*"
    return history, render_speed(note="Stopped, so no timings."), *finish_question()


def switch_model(model_key):
    yield (f"Loading {model_label(model_key)}. Large models can take several minutes.",
           gr.update(interactive=False), gr.update(interactive=False))
    try:
        load_model(model_key)
        status = f"{model_label(model_key)} is ready."
    except Exception as e:
        status = f"{model_label(model_key)} could not be loaded: {e}"
    yield status, gr.update(interactive=True), gr.update(interactive=_engine is not None)


# ---------------------------------------------------------------------------
# Look: the colours of a printed nautical chart. White is deep water, blue is
# shallow water (the conversation), buff is land (what comes from the guide),
# magenta marks what you can press, as it marks navigation aids on a chart.
# ---------------------------------------------------------------------------
magenta = gr.themes.Color(
    c50="#FBEFF6", c100="#F6DCEB", c200="#EDB9D7", c300="#E08DBD", c400="#CC5C9C", c500="#B93A83",
    c600="#A8246F", c700="#8A1B5A", c800="#6E1648", c900="#521136", c950="#360A23",
)
ink = gr.themes.Color(
    c50="#F5F9FB", c100="#E6EFF4", c200="#CADCE6", c300="#A5C0D0", c400="#6F93AA", c500="#4B7089",
    c600="#36566D", c700="#274257", c800="#1B3246", c900="#14283C", c950="#0C1A29",
)
theme = gr.themes.Base(
    primary_hue=magenta,
    secondary_hue=ink,
    neutral_hue=ink,
    font=[gr.themes.GoogleFont("Archivo"), "Segoe UI", "system-ui", "sans-serif"],
    radius_size=gr.themes.sizes.radius_sm,
).set(
    body_background_fill="#FFFFFF",
    body_background_fill_dark="#0C1A29",
    body_text_color="#14283C",
    body_text_color_dark="#E6EFF4",
    block_background_fill="#FFFFFF",
    block_background_fill_dark="#12263A",
    block_border_color="#A5C0D0",
    block_border_color_dark="#36566D",
    block_shadow="none",
    block_shadow_dark="none",
    input_background_fill="#FFFFFF",
    input_background_fill_dark="#0C1A29",
    input_border_color="#6F93AA",
    input_border_color_dark="#4B7089",
    input_border_color_focus="#A8246F",
    input_border_color_focus_dark="#E08DBD",
    button_primary_background_fill="#A8246F",
    button_primary_background_fill_hover="#8A1B5A",
    button_primary_background_fill_dark="#CC5C9C",
    button_primary_background_fill_hover_dark="#E08DBD",
    button_primary_text_color="#FFFFFF",
    button_primary_text_color_dark="#0C1A29",
    button_secondary_background_fill="#FFFFFF",
    button_secondary_background_fill_dark="#12263A",
    button_secondary_border_color="#6F93AA",
    button_secondary_border_color_dark="#4B7089",
    button_secondary_text_color="#14283C",
    button_secondary_text_color_dark="#E6EFF4",
)

CSS = """
.gradio-container {
    --mc-water: #D3E9F2; --mc-land: #F1E3B4; --mc-land-line: #C9B26A; --mc-contour: #3F7FA3;
    --mc-ink: #14283C; --mc-ink-soft: #36566D; --mc-mark: #A8246F;
    max-width: 1180px !important; margin: 0 auto !important;
}
.dark .gradio-container, .gradio-container.dark {
    --mc-water: #1B3F55; --mc-land: #3B3423; --mc-land-line: #7A6A3A; --mc-contour: #6FA9C9;
    --mc-ink: #E6EFF4; --mc-ink-soft: #A5C0D0; --mc-mark: #E08DBD;
}
#mc-header { border-bottom: 2px solid var(--mc-contour); padding-bottom: 14px; align-items: end; }
#mc-header h1 { font-size: 2.25rem; font-weight: 700; font-style: italic; letter-spacing: -0.01em;
    line-height: 1.1; margin: 0 0 6px; color: var(--mc-ink); }
#mc-header p { font-size: 1rem; line-height: 1.45; max-width: 62ch; margin: 0; color: var(--mc-ink-soft); }
#mc-status { min-height: 1.4em; }
#mc-status p { font-size: 0.875rem; color: var(--mc-ink-soft); margin: 4px 0 0; }

#mc-chat { border: 1px solid var(--mc-contour); }
#mc-chat .message { font-size: 1rem; line-height: 1.5; }
#mc-chat .user { background: var(--mc-water); border-color: var(--mc-water); color: var(--mc-ink); }
#mc-chat .bot { background: transparent; border-color: transparent; color: var(--mc-ink); }

.mc-side, #mc-speed-block { gap: 4px !important; }
.mc-side h2 { font-size: 1.125rem; font-weight: 600; margin: 0 0 8px; color: var(--mc-ink); }
.mc-note { font-size: 0.9375rem; line-height: 1.45; color: var(--mc-ink-soft); margin: 0; }
.mc-warning { background: var(--mc-land); border-left: 4px solid var(--mc-mark); padding: 10px 12px;
    color: var(--mc-ink); }
.mc-passages { list-style: none; margin: 0; padding: 0; display: grid; gap: 10px;
    max-height: 420px; overflow-y: auto; }
.mc-passage { background: var(--mc-land); border: 1px solid var(--mc-land-line); padding: 10px 12px; }
.mc-passage-head { display: flex; justify-content: space-between; align-items: baseline; gap: 12px;
    margin-bottom: 6px; }
.mc-page { font-weight: 700; color: var(--mc-ink); }
.mc-score { font-size: 0.8125rem; color: var(--mc-ink-soft); font-variant-numeric: tabular-nums; }
.mc-passage p { font-size: 0.875rem; line-height: 1.5; margin: 0; color: var(--mc-ink);
    white-space: pre-wrap; overflow-wrap: anywhere; }
.mc-speed { display: grid; grid-template-columns: 1fr 1fr; gap: 10px 16px; margin: 0; }
.mc-speed dt { font-size: 0.8125rem; color: var(--mc-ink-soft); }
.mc-speed dd { font-size: 1.125rem; font-weight: 600; margin: 0; color: var(--mc-ink);
    font-variant-numeric: tabular-nums; }
#mc-speed-block { border-top: 1px solid var(--mc-contour); padding-top: 14px; margin-top: 6px; }
button:focus-visible, textarea:focus-visible, input:focus-visible { outline: 2px solid var(--mc-mark);
    outline-offset: 2px; }
"""


def build_demo(models, selected):
    with gr.Blocks(title="MarineChat") as demo:
        with gr.Row(elem_id="mc-header"):
            with gr.Column(scale=3):
                gr.HTML(
                    "<h1>MarineChat</h1>"
                    "<p>Ask about Italian ports: winds, depths, VHF channels, contacts. "
                    "Answers are written on this computer's CPU from the ports guide loaded on it.</p>"
                )
            with gr.Column(scale=2, min_width=260):
                model = gr.Dropdown(
                    choices=[(model_label(key), key) for key in models], value=selected,
                    label="Model", info="Bigger models answer better and more slowly.",
                    interactive=len(models) > 1, filterable=False,
                )
                status = gr.Markdown(f"{model_label(selected)} is ready.", elem_id="mc-status")

        with gr.Row():
            with gr.Column(scale=3, min_width=340):
                chat = gr.Chatbot(
                    elem_id="mc-chat", height=480, show_label=False, layout="bubble", buttons=["copy"],
                    placeholder="Ask a question below, or pick one of the examples.",
                )
                with gr.Row():
                    box = gr.Textbox(
                        show_label=False, placeholder="For example: which winds are dangerous in the port of Genoa?",
                        scale=5, lines=1, max_lines=4, autofocus=True, container=False,
                    )
                    ask = gr.Button("Ask", variant="primary", scale=1, min_width=96)
                    stop = gr.Button("Stop", variant="secondary", scale=1, min_width=96, visible=False)
                gr.Examples(examples=EXAMPLES, inputs=box, label="Example questions")
            with gr.Column(scale=2, min_width=300, elem_classes="mc-side"):
                gr.HTML("<h2>Passages used</h2>")
                sources = gr.HTML(render_sources())
                with gr.Column(elem_id="mc-speed-block"):
                    gr.HTML("<h2>Speed</h2>")
                    speed = gr.HTML(render_speed())

        question = gr.State("")
        buttons = [ask, stop, model]

        answering = []
        for trigger in (box.submit, ask.click):
            started = trigger(start_question, [box, chat], [box, chat, question, sources, speed, *buttons])
            answered = started.success(answer_question, [question, chat], [chat, sources, speed])
            answered.then(finish_question, None, buttons)
            answering.append(answered)
        stop.click(stop_answer, [chat], [chat, speed, *buttons], cancels=answering)

        model.change(switch_model, [model], [status, model, ask])
    return demo


if __name__ == "__main__":
    models = available_models()
    if not models:
        raise SystemExit("No exported model found in model/. Run download_model.py first.")
    parser = argparse.ArgumentParser(description="MarineChat Gradio UI")
    parser.add_argument("--model", choices=models, default=default_model if default_model in models else models[0],
                        help="Model loaded at start (it can be changed in the page)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser tab at start")
    args = parser.parse_args()

    load_model(args.model)
    build_demo(models, args.model).launch(inbrowser=not args.no_browser, theme=theme, css=CSS)
