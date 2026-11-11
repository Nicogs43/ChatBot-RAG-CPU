from pathlib import Path

import config


def test_paths_are_absolute_and_inside_the_repo():
    assert config.ROOT.is_absolute()
    assert (config.ROOT / "bin" / "config.py").exists()
    for path in (config.vectorstore_path, config.pdf_path, config.reranker_path):
        assert Path(path).is_absolute()
        assert config.ROOT in Path(path).parents


def test_every_model_has_a_path_under_model_dir_and_a_system_prompt():
    for key, details in config.model_dict.items():
        assert config.ROOT / "model" in Path(details["model_path"]).parents, key
        assert details["system_prompt"].strip(), key


def test_default_model_is_a_known_key():
    assert config.default_model in config.model_dict


def test_user_prompt_template_has_question_and_context():
    text = config.USER_PROMPT_TEMPLATE.format(input="Q?", context="C.")
    assert "Q?" in text and "C." in text


def test_rerank_keeps_no_more_than_search_returns():
    assert config.retrieval_config["vector_rerank_top_n"] <= config.retrieval_config["vector_search_top_k"]


def test_model_label_shows_name_and_variant():
    assert config.model_label("qwen_0.5b") == "Qwen2.5-0.5B-Instruct"
    assert config.model_label("phi_int4") == "Phi-3.5-mini-instruct (int4)"


def test_available_models_lists_only_exports_on_disk(tmp_path, monkeypatch):
    present = tmp_path / "present"
    present.mkdir()
    (present / "openvino_model.xml").write_text("<net/>")
    monkeypatch.setattr(
        config,
        "model_dict",
        {
            "here": {"model_path": str(present), "system_prompt": "x"},
            "missing": {"model_path": str(tmp_path / "missing"), "system_prompt": "x"},
        },
    )
    assert config.available_models() == ["here"]
