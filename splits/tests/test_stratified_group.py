import random

from splits.schema import NormalizedRow, SPLIT_TEST, SPLIT_TRAIN, SPLIT_VAL
from splits.stratified_group import stratified_grouped_three_way


def _make_rows(n_speakers=40, emotions=("angry", "happy", "sad", "neutral", "fearful"),
               per_speaker=10, seed=0):
    rng = random.Random(seed)
    rows: list[NormalizedRow] = []
    for s in range(n_speakers):
        for u in range(per_speaker):
            emo = rng.choice(emotions)
            rows.append(
                NormalizedRow(
                    dataset="batch01",
                    row_id=f"s{s:03d}_u{u:03d}",
                    source_index=len(rows),
                    speaker_id=f"spk_{s:03d}",
                    emotion=emo,
                    gender=rng.choice(["M", "F"]),
                    age_category=rng.choice(["child", "adult"]),
                    augmented=False,
                )
            )
    return rows


def test_split_disjoint_and_covers_all():
    rows = _make_rows()
    out = stratified_grouped_three_way(
        rows,
        ratios={SPLIT_TRAIN: 0.7, SPLIT_VAL: 0.15, SPLIT_TEST: 0.15},
        stratify_by="emotion",
        group_by_field="speaker_id",
        seed=42,
    )
    indices = set(out[SPLIT_TRAIN]) | set(out[SPLIT_VAL]) | set(out[SPLIT_TEST])
    assert len(indices) == len(rows)
    assert set(out[SPLIT_TRAIN]).isdisjoint(out[SPLIT_VAL])
    assert set(out[SPLIT_TRAIN]).isdisjoint(out[SPLIT_TEST])
    assert set(out[SPLIT_VAL]).isdisjoint(out[SPLIT_TEST])


def test_speaker_disjoint():
    rows = _make_rows()
    out = stratified_grouped_three_way(
        rows,
        ratios={SPLIT_TRAIN: 0.7, SPLIT_VAL: 0.15, SPLIT_TEST: 0.15},
        stratify_by="emotion",
        group_by_field="speaker_id",
        seed=42,
    )
    def speakers(idxs):
        return {rows[i]["speaker_id"] for i in idxs}
    t = speakers(out[SPLIT_TRAIN])
    v = speakers(out[SPLIT_VAL])
    s = speakers(out[SPLIT_TEST])
    assert t.isdisjoint(v)
    assert t.isdisjoint(s)
    assert v.isdisjoint(s)


def test_ratios_within_tolerance():
    rows = _make_rows(n_speakers=60, per_speaker=20)
    out = stratified_grouped_three_way(
        rows,
        ratios={SPLIT_TRAIN: 0.7, SPLIT_VAL: 0.15, SPLIT_TEST: 0.15},
        stratify_by="emotion",
        group_by_field="speaker_id",
        seed=7,
    )
    n = len(rows)
    for split, target in [(SPLIT_TRAIN, 0.70), (SPLIT_VAL, 0.15), (SPLIT_TEST, 0.15)]:
        frac = len(out[split]) / n
        assert abs(frac - target) < 0.05, f"{split}: {frac:.3f} vs target {target:.3f}"


def test_all_emotions_present():
    rows = _make_rows()
    out = stratified_grouped_three_way(
        rows,
        ratios={SPLIT_TRAIN: 0.7, SPLIT_VAL: 0.15, SPLIT_TEST: 0.15},
        stratify_by="emotion",
        group_by_field="speaker_id",
        seed=42,
    )
    for split in [SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST]:
        emos = {rows[i]["emotion"] for i in out[split]}
        assert len(emos) >= 3  # at least a decent coverage


def test_deterministic():
    rows = _make_rows()
    a = stratified_grouped_three_way(
        rows, {"train": 0.7, "val": 0.15, "test": 0.15},
        "emotion", "speaker_id", seed=123,
    )
    b = stratified_grouped_three_way(
        rows, {"train": 0.7, "val": 0.15, "test": 0.15},
        "emotion", "speaker_id", seed=123,
    )
    assert a == b


def test_asymmetric_ratios():
    rows = _make_rows(n_speakers=100, per_speaker=20)
    out = stratified_grouped_three_way(
        rows,
        ratios={SPLIT_TRAIN: 0.9, SPLIT_VAL: 0.05, SPLIT_TEST: 0.05},
        stratify_by="emotion",
        group_by_field="speaker_id",
        seed=1,
    )
    n = len(rows)
    train_frac = len(out[SPLIT_TRAIN]) / n
    # 90/5/5 → train ~0.90; accept up to ±0.06
    assert 0.84 <= train_frac <= 0.96, train_frac
