import pytest

from routemind.data import assert_disjoint, load_records, split_records, write_jsonl
from routemind.prompting import pad_features, training_example
from routemind.training import TrainConfig, prepare_records


class TokenizerStub:
    eos_token_id = 99

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": True, "add_generation_prompt": True, "enable_thinking": False,
                          "return_dict": False}
        assert len(messages) == 2
        return [10, 11, 12]

    def encode(self, text, **kwargs):
        assert kwargs == {"add_special_tokens": False}
        return [20, 21, 22, 23]


def test_grouped_split_is_reproducible_and_disjoint(records):
    first, second = split_records(records), split_records(list(reversed(records)))
    for key in first:
        assert {r.id for r in first[key]} == {r.id for r in second[key]}
    assert_disjoint(*first.values())
    for group in {r.scenario_group_id for r in records}:
        assert sum(any(r.scenario_group_id == group for r in rows) for rows in first.values()) == 1


def test_cross_split_leakage_detected(records):
    with pytest.raises(ValueError, match="leakage"):
        assert_disjoint([records[0]], [records[1]])  # translations of one scenario


def test_duplicate_text_is_rejected(records, tmp_path):
    duplicate = records[0].model_copy(deep=True)
    duplicate.id = "another-id"
    write_jsonl(tmp_path / "duplicates.jsonl", [records[0], duplicate])
    with pytest.raises(ValueError, match="duplicate"):
        load_records(tmp_path / "duplicates.jsonl")


def test_prompt_and_padding_are_not_supervised(records):
    row = records[0]
    feature = training_example(TokenizerStub(), row.input, row.target, 20)
    assert feature["labels"] == [-100, -100, -100, 20, 21, 22, 23, 99]
    padded = pad_features([feature, {"input_ids": [5, 99], "attention_mask": [1, 1], "labels": [5, 99]}], 99)
    assert padded["labels"][0][-1] == 99  # Real EOS stays supervised, even when pad==EOS.
    assert padded["labels"][1][2:] == [-100] * 6
    assert padded["attention_mask"][1][2:] == [0] * 6


def test_overlength_target_is_not_silently_truncated(records):
    with pytest.raises(ValueError, match="no silent truncation"):
        training_example(TokenizerStub(), records[0].input, records[0].target, 4)


def test_unreviewed_training_requires_explicit_override(records, tmp_path):
    splits = split_records(records)
    for name, rows in splits.items():
        write_jsonl(tmp_path / f"{name}.jsonl", rows)
    config = TrainConfig(train_file=str(tmp_path / "train.jsonl"),
                         validation_file=str(tmp_path / "validation.jsonl"),
                         test_file=str(tmp_path / "test.jsonl"))
    with pytest.raises(ValueError, match="approved"):
        prepare_records(config)
    assert len(prepare_records(config, True)) == 3
