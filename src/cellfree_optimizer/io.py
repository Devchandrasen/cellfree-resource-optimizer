"""Portable network and result files with explicit indexing and provenance."""

from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import sys
import numpy as np
from scipy.io import savemat

from .config import NetworkConfig
from .network import Network


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(jsonable(value), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def environment():
    return dict(
        python=sys.version.split()[0],
        platform=platform.platform(),
        packages={
            name: importlib.metadata.version(name)
            for name in [
                "cellfree-resource-optimizer",
                "numpy",
                "scipy",
                "clarabel",
                "threadpoolctl",
            ]
        },
    )


def save_network(network, path):
    np.savez_compressed(
        path,
        config_json=json.dumps(network.config.to_dict()),
        ap_positions=network.ap_positions,
        user_positions=network.user_positions,
        **network.moments,
    )


def load_network(path):
    with np.load(path, allow_pickle=False) as data:
        config = NetworkConfig.from_dict(json.loads(str(data["config_json"])))
        moments = {
            name: data[name].copy()
            for name in data.files
            if name not in {"config_json", "ap_positions", "user_positions"}
        }
        for key in ["M", "N", "K", "Dmax"]:
            moments[key] = int(moments[key])
        moments["rho"] = float(moments["rho"])
        return Network(config, moments, data["ap_positions"].copy(), data["user_positions"].copy())


def save_result(network, result, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    save_network(network, folder / "network.npz")
    np.savez_compressed(folder / "solution.npz", **result.solution)
    net = dict(network.moments)
    net["order"] = net["order"] + 1
    net["rank"] = net["rank"] + 1
    savemat(
        folder / "validation.mat", {"net": net, "solution": result.solution}, do_compression=True
    )
    write_json(
        folder / "result.json",
        dict(
            schema_version=1,
            method=result.method,
            service=result.service,
            config=network.config.to_dict(),
            coherent_users=int(np.sum(result.solution["z"])),
            metadata=result.metadata,
            validation=result.validation,
            environment=environment(),
            files={
                name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
                for name in ["network.npz", "solution.npz", "validation.mat"]
            },
        ),
    )
