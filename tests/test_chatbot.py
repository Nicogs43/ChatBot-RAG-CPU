import pytest
from conftest import FakeModel, FakeTokenizer

import chatbot
from config import USER_PROMPT_TEMPLATE, generation_kwargs


def test_unknown_model_key_is_rejected_before_loading_anything(monkeypatch):
    monkeypatch.setattr(chatbot, "load_openvino_model", lambda path: pytest.fail("must not load"))
    with pytest.raises(KeyError, match="Unknown model"):
        chatbot.RagEngine("not_a_model", vectorstore=object())


def test_messages_carry_system_prompt_question_and_all_passages(make_engine, docs):
    engine, _, _ = make_engine()
    system, user = engine.build_messages("What VHF channel?", docs)
    assert system == {"role": "system", "content": engine.system_prompt}
    assert user["role"] == "user"
    context = "\n\n".join(d.page_content for d in docs)
    assert user["content"] == USER_PROMPT_TEMPLATE.format(input="What VHF channel?", context=context)


def test_prompt_uses_the_tokenizer_chat_template(make_engine, docs):
    engine, _, _ = make_engine()
    prompt = engine.build_prompt("What VHF channel?", docs)
    assert prompt.startswith("<system>") and prompt.endswith("<assistant>")


def test_template_without_system_role_gets_system_prompt_in_the_user_turn(make_engine, docs):
    engine, _, tokenizer = make_engine(tokenizer=FakeTokenizer(supports_system=False))
    prompt = engine.build_prompt("What VHF channel?", docs)
    assert [m["role"] for m in tokenizer.seen_messages] == ["user"]
    assert engine.system_prompt in prompt and "What VHF channel?" in prompt


def test_stream_yields_the_answer_piece_by_piece(make_engine, docs):
    engine, model, _ = make_engine()
    assert list(engine.stream("q", docs)) == ["Channel ", "12 ", "and ", "16."]
    assert {k: model.kwargs[k] for k in generation_kwargs} == generation_kwargs


def test_stream_retrieves_when_no_documents_are_given(make_engine):
    engine, _, tokenizer = make_engine()
    "".join(engine.stream("q"))
    assert "Imperia" in tokenizer.seen_messages[-1]["content"]


def test_invoke_returns_input_context_and_stripped_answer(make_engine, docs):
    engine, _, _ = make_engine(model=FakeModel(words=("  Channel ", "12.  ")))
    assert engine.invoke("q") == {"input": "q", "context": docs, "answer": "Channel 12."}


def test_invoke_works_with_no_retrieved_documents(make_engine):
    engine, _, _ = make_engine(retrieved=[])
    result = engine.invoke("q")
    assert result["context"] == [] and result["answer"]


def test_generation_error_is_raised_to_the_caller(make_engine, docs):
    engine, _, _ = make_engine(model=FakeModel(error=RuntimeError("out of memory")))
    with pytest.raises(RuntimeError, match="out of memory"):
        list(engine.stream("q", docs))


def test_closing_the_stream_early_stops_the_model(make_engine, docs):
    model = FakeModel(words=tuple(f"w{i} " for i in range(100_000)))
    engine, _, _ = make_engine(model=model)
    stream = engine.stream("q", docs)
    assert next(stream) == "w0 "
    stream.close()  # what Gradio does when Stop is pressed
    assert model.words_sent < 100_000


def test_engine_answers_again_after_an_early_stop(make_engine, docs):
    engine, model, _ = make_engine()
    stream = engine.stream("q", docs)
    next(stream)
    stream.close()
    assert "".join(engine.stream("q", docs)) == "Channel 12 and 16."
    assert model.calls == 2


def test_cancel_before_a_question_does_not_block_the_next_answer(make_engine, docs):
    engine, _, _ = make_engine()
    engine.cancel()
    assert "".join(engine.stream("q", docs)) == "Channel 12 and 16."
