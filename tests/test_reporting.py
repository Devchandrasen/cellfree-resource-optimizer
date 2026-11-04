import json
import numpy as np
import pytest
from cellfree_optimizer.cli import summarize


def test_summary_uses_layout_blocks_and_paired_ratios(tmp_path):
    manifest = dict(status="complete", layouts=2, seeds=2, methods=["joint", "strong-cjt"], runs=[])
    candidate = [[2.0, 6.0], [3.0, 8.0]]
    baseline = [[1.0, 4.0], [2.0, 4.0]]
    for layout in range(2):
        for seed in range(2):
            for method, rates in [("joint", candidate), ("strong-cjt", baseline)]:
                relative = f"layout_{layout}/seed_{seed}/{method}"
                folder = tmp_path / relative
                folder.mkdir(parents=True)
                (folder / "result.json").write_text(
                    json.dumps(
                        dict(service=rates[layout][seed], config=dict(fronthaul=4, max_aps=2))
                    ),
                    encoding="utf-8",
                )
                manifest["runs"].append(
                    dict(condition=0, layout=layout, seed=seed, method=method, path=relative)
                )
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    rows = summarize(tmp_path)
    joint = next(row for row in rows if row["method"] == "joint")
    expected = 100 * np.mean(np.asarray(candidate) / baseline - 1)
    assert joint["paired_gain_percent"] == pytest.approx(expected)
    assert joint["paired_gain_percent"] != pytest.approx(
        100 * (np.mean(candidate) / np.mean(baseline) - 1)
    )
    assert joint["paired_layouts"] == 2
    assert joint["runs"] == 4
    assert joint["pointwise_ci95_low"] == pytest.approx(expected)
    assert joint["pointwise_ci95_high"] == pytest.approx(expected)


def test_incomplete_batch_is_not_reported_as_complete(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps(dict(status="running")), encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete"):
        summarize(tmp_path)
