import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from scipy.io import loadmat
from cellfree_optimizer import NetworkConfig, generate_network, optimize
from cellfree_optimizer.io import save_result


def test_saved_solution_can_be_validated_in_new_process(tmp_path):
    n = generate_network(NetworkConfig(aps=3, users=2, antennas=2, max_aps=2, pilot_length=2))
    r = optimize(n, "strong-cjt", seconds=1, seed=0)
    folder = tmp_path / "run"
    save_result(n, r, folder)
    done = subprocess.run(
        [sys.executable, "-m", "cellfree_optimizer", "validate", str(folder)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(done.stdout)["feasible"]
    mat = loadmat(folder / "validation.mat", simplify_cells=True)
    np.testing.assert_array_equal(
        np.asarray(mat["net"]["order"]).reshape(3, 2), n.moments["order"] + 1
    )
    saved = json.loads((folder / "result.json").read_text())
    assert saved["service"] == r.service
    assert len(saved["files"]) == 3


def test_cli_does_not_overwrite_existing_results(tmp_path):
    done = subprocess.run(
        [sys.executable, "-m", "cellfree_optimizer", "run", "--output", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert done.returncode == 2
    assert "already exists" in done.stderr
    assert list(Path(tmp_path).iterdir()) == []
