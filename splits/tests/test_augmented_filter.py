from splits.augmented_filter import extract_augmented_flag, is_augmented


def test_top_level_bool():
    assert extract_augmented_flag({"augmented": True}) is True
    assert extract_augmented_flag({"augmented": False}) is False
    assert extract_augmented_flag({"is_augmented": True}) is True


def test_top_level_stringy():
    assert extract_augmented_flag({"augmented": "true"}) is True
    assert extract_augmented_flag({"augmented": "False"}) is False
    assert extract_augmented_flag({"augmented": 1}) is True
    assert extract_augmented_flag({"augmented": 0}) is False


def test_missing():
    assert extract_augmented_flag({}) is None
    assert extract_augmented_flag({"emotion": "angry"}) is None


def test_metadata_dict():
    assert extract_augmented_flag({"metadata": {"augmented": True}}) is True
    assert extract_augmented_flag({"metadata": {"augmented": False}}) is False


def test_metadata_json_string():
    assert extract_augmented_flag({"metadata": '{"augmented": true}'}) is True
    assert extract_augmented_flag({"metadata": '{"augmented": false}'}) is False
    assert extract_augmented_flag({"metadata": "{}"}) is None


def test_metadata_pseudo_json():
    assert extract_augmented_flag({"metadata": "augmented=true,foo=bar"}) is True
    assert extract_augmented_flag({"metadata": "augmented=false"}) is False


def test_top_level_wins_over_metadata():
    row = {"augmented": False, "metadata": {"augmented": True}}
    assert extract_augmented_flag(row) is False


def test_is_augmented_only_true_on_explicit_true():
    assert is_augmented({"augmented": True}) is True
    assert is_augmented({"augmented": False}) is False
    assert is_augmented({}) is False
