from splits.kazemo_speakers import (
    kazemo_three_way_split,
    parse_kazemo_speaker,
    resolve_kazemo_speakers,
)


def test_parse_from_text_prefix():
    row = {"text": "aqtolkyn_happy_0102|some transcription"}
    assert parse_kazemo_speaker(row) == "aqtolkyn"


def test_parse_from_audio_path():
    row = {"audio": {"path": "/x/y/Mamyr_angry_001.wav"}}
    assert parse_kazemo_speaker(row) == "mamyr"


def test_parse_multitoken_speaker():
    row = {"text": "speaker_X_happy_utt_2|t"}
    assert parse_kazemo_speaker(row) == "speaker_x"


def test_parse_no_emotion_returns_none():
    row = {"text": "random_stuff|t", "audio": {"path": "noclue.wav"}}
    assert parse_kazemo_speaker(row) is None


def test_parse_fearful_synonym():
    # _EMO_MAP includes "scared" → fearful
    row = {"text": "voiceA_scared_7|t"}
    assert parse_kazemo_speaker(row) == "voicea"


def test_parse_emokaz_utterance_first_layout():
    # EmoKaz.zip uses <utt_id>_<emotion>_<narrator>.wav — narrator-token wins.
    row = {"audio": {"path": "/cache/extracted_partial/1263201035_neutral_F1.wav"}}
    assert parse_kazemo_speaker(row) == "f1"


def test_parse_emokaz_narrator_dir():
    # Or speaker may live in a parent dir like F1/neutral_001.wav
    row = {"audio": {"path": "/cache/EmoKaz/M2/neutral_001.wav"}}
    assert parse_kazemo_speaker(row) == "m2"


def test_three_way_split_deterministic():
    rows = []
    for i in range(10):
        rows.append({"speaker_id": "a"})
    for i in range(10):
        rows.append({"speaker_id": "b"})
    for i in range(20):
        rows.append({"speaker_id": "c"})
    out = kazemo_three_way_split(
        rows,
        train_speakers={"a", "b"},
        valtest_speakers={"c"},
        valtest_ratio={"val": 0.5, "test": 0.5},
        seed=42,
    )
    assert len(out["train"]) == 20
    assert len(out["val"]) == 10
    assert len(out["test"]) == 10
    # No overlap
    assert set(out["val"]).isdisjoint(out["test"])
    # All train indices correspond to speakers a/b
    for i in out["train"]:
        assert rows[i]["speaker_id"] in {"a", "b"}
    for i in out["val"] + out["test"]:
        assert rows[i]["speaker_id"] == "c"


def test_three_way_split_is_seeded():
    rows = [{"speaker_id": "c"} for _ in range(100)]
    a = kazemo_three_way_split(rows, set(), {"c"}, {"val": 0.5, "test": 0.5}, seed=1)
    b = kazemo_three_way_split(rows, set(), {"c"}, {"val": 0.5, "test": 0.5}, seed=1)
    assert a == b
    c = kazemo_three_way_split(rows, set(), {"c"}, {"val": 0.5, "test": 0.5}, seed=2)
    assert a != c


def test_resolve_by_index():
    train, valtest = resolve_kazemo_speakers(
        ["c", "a", "b"], strategy="by_index", train_indices=[0, 1], valtest_index=2
    )
    # sorted → [a, b, c]; train = {a,b}, valtest = {c}
    assert train == {"a", "b"}
    assert valtest == {"c"}


def test_resolve_by_name():
    train, valtest = resolve_kazemo_speakers(
        ["a", "b", "c"], strategy="by_name", train_names=["a", "c"], valtest_name="b"
    )
    assert train == {"a", "c"}
    assert valtest == {"b"}


def test_resolve_by_index_out_of_range():
    try:
        resolve_kazemo_speakers(
            ["a", "b"], strategy="by_index", train_indices=[0], valtest_index=5
        )
    except ValueError:
        return
    raise AssertionError("expected ValueError")
