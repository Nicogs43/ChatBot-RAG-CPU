import json

import pandas as pd

import proc_data

LEGACY_ANSWER = (
    "<|im_start|>system\nYou are an assistant.<|im_end|>\n"
    "<|im_start|>user\nQuestion: What winds? \nContext: Genoa has northerly winds. \n<|im_end|>\n"
    "<|im_start|>assistant<|im_end|>\nMostly northerly."
)


def test_legacy_answer_is_split_into_answer_and_context():
    answer, context = proc_data._split_legacy_answer(LEGACY_ANSWER)
    assert answer == "Mostly northerly."
    assert context == ["Genoa has northerly winds."]


def test_answer_without_chat_markers_is_kept_as_is():
    assert proc_data._split_legacy_answer("Mostly northerly.") == ("Mostly northerly.", [""])


def test_both_csv_formats_become_the_same_json_shape(tmp_path, monkeypatch):
    monkeypatch.setattr(proc_data, "DATA_DIR", tmp_path)
    monkeypatch.setattr(proc_data, "model_dict", {"old": {}, "new": {}})
    pd.DataFrame({"question": ["What winds?"], "answer": [LEGACY_ANSWER], "ground_truths": ["Northerly."]}).to_csv(
        tmp_path / "old_data_with_answer.csv", index=False
    )
    pd.DataFrame(
        {
            "question": ["What winds?"],
            "answer": ["Mostly northerly."],
            "ground_truths": ["Northerly."],
            "contexts": [json.dumps(["Genoa has northerly winds.", "Second passage."])],
        }
    ).to_csv(tmp_path / "new_data_with_answer.csv", index=False)

    proc_data.proc_datacreated_by_qwen()

    old = json.loads((tmp_path / "old_with_answer_and_contexts_cleaned.json").read_text())
    new = json.loads((tmp_path / "new_with_answer_and_contexts_cleaned.json").read_text())
    assert old == {
        "question": ["What winds?"],
        "ground_truth": ["Northerly."],
        "answer": ["Mostly northerly."],
        "contexts": [["Genoa has northerly winds."]],
    }
    assert new["answer"] == ["Mostly northerly."]
    assert new["contexts"] == [["Genoa has northerly winds.", "Second passage."]]
