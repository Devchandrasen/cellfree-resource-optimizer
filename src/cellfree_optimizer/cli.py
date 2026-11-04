"""Command-line entry points for runs, matched batches and validation."""

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import sys
import numpy as np

from . import NetworkConfig, generate_network, optimize, validate_solution
from .io import environment, load_network, save_result, write_json
from .optimizer import METHODS


def configuration(path):
    return (
        NetworkConfig.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
        if path
        else NetworkConfig()
    )


def run(args):
    if Path(args.output).exists():
        raise ValueError("Output already exists. Choose a new output directory.")
    network = (
        load_network(args.network) if args.network else generate_network(configuration(args.config))
    )
    result = optimize(network, args.method, args.seconds, args.seed)
    save_result(network, result, args.output)
    if args.plot:
        from .visualization import plot_result

        plot_result(network, result, args.output)
    print(f"{result.method}: minimum service {result.service:.6f} bit/s/Hz")
    print(
        f"Accepted at {result.metadata['accepted_point_seconds']:.3f} s, wall time {result.metadata['wall_seconds']:.3f} s"
    )
    print(
        f"Feasible: {result.validation['feasible']}, maximum residual {result.validation['max_residual']:.3e}"
    )
    print(f"Saved: {args.output}")


def benchmark(args):
    config = configuration(args.config)
    methods = args.methods.split(",")
    if len(methods) != len(set(methods)) or any(m not in METHODS for m in methods):
        raise ValueError("methods must be unique supported method names")
    if args.layouts < 1 or args.seeds < 1:
        raise ValueError("layouts and seeds must be positive")
    capacities = (
        [float(v) for v in args.fronthaul.split(",")] if args.fronthaul else [config.fronthaul]
    )
    degrees = [int(v) for v in args.max_aps.split(",")] if args.max_aps else [config.max_aps]
    configurations = [replace(config, fronthaul=c, max_aps=d) for c in capacities for d in degrees]
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    manifest = dict(
        schema_version=1,
        status="running",
        base_config=config.to_dict(),
        methods=methods,
        seconds=args.seconds,
        layouts=args.layouts,
        seeds=args.seeds,
        method_order="deterministic shuffle per matched block",
        environment=environment(),
        runs=[],
    )
    write_json(root / "manifest.json", manifest)
    for condition, cfg in enumerate(configurations):
        for layout in range(args.layouts):
            network = generate_network(replace(cfg, seed=config.seed + layout))
            for seed in range(args.seeds):
                order = np.random.default_rng(7919 * (layout + 1) + seed).permutation(methods)
                for method in order:
                    method = str(method)
                    folder = f"condition_{condition}/layout_{layout}/seed_{seed}/{method}"
                    result = optimize(network, method, args.seconds, seed)
                    save_result(network, result, root / folder)
                    manifest["runs"].append(
                        dict(
                            condition=condition,
                            layout=layout,
                            seed=seed,
                            method=method,
                            path=folder,
                        )
                    )
                    write_json(root / "manifest.json", manifest)
                    print(
                        f"condition {condition}, layout {layout}, seed {seed}, {method}: {result.service:.6f}",
                        flush=True,
                    )
    manifest["status"] = "complete"
    write_json(root / "manifest.json", manifest)
    summarize(root)


def summarize(root):
    from scipy.stats import t as student

    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete":
        raise ValueError("The batch is incomplete. A partial batch is not summarized as complete.")
    rows, data = [], {}
    for entry in manifest["runs"]:
        r = json.loads((root / entry["path"] / "result.json").read_text(encoding="utf-8"))
        data[(entry["condition"], entry["layout"], entry["seed"], entry["method"])] = r
    for condition in sorted({key[0] for key in data}):
        for method in manifest["methods"]:
            values = [r for key, r in data.items() if key[0] == condition and key[3] == method]
            gains = []
            for layout in range(manifest["layouts"]):
                paired = []
                for seed in range(manifest["seeds"]):
                    candidate = data.get((condition, layout, seed, method))
                    baseline = data.get((condition, layout, seed, "strong-cjt"))
                    if candidate and baseline and baseline["service"] > 0:
                        paired.append(100 * (candidate["service"] / baseline["service"] - 1))
                if len(paired) == manifest["seeds"]:
                    gains.append(float(np.mean(paired)))
            mean = float(np.mean(gains)) if gains else None
            half = (
                float(
                    student.ppf(0.975, len(gains) - 1) * np.std(gains, ddof=1) / np.sqrt(len(gains))
                )
                if len(gains) > 1
                else None
            )
            rows.append(
                dict(
                    condition=condition,
                    mean_fronthaul=values[0]["config"]["fronthaul"],
                    max_aps=values[0]["config"]["max_aps"],
                    method=method,
                    layouts=manifest["layouts"],
                    runs=len(values),
                    mean_service=float(np.mean([r["service"] for r in values])),
                    paired_layouts=len(gains),
                    paired_gain_percent=mean,
                    pointwise_ci95_low=mean - half if half is not None else None,
                    pointwise_ci95_high=mean + half if half is not None else None,
                )
            )
    with (root / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(root / "summary.json", rows)
    print(f"Summary: {root / 'summary.csv'}")
    return rows


def validate(args):
    folder = Path(args.folder)
    network = load_network(folder / "network.npz")
    with np.load(folder / "solution.npz", allow_pickle=False) as data:
        solution = {name: data[name].copy() for name in data.files}
    result = validate_solution(network, solution)
    print(json.dumps(result, indent=2))
    if not result["feasible"]:
        raise ValueError("The saved solution violates the model constraints")


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="cellfree", description="Cell-free wireless resource optimizer"
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run", help="Optimize one generated or saved network")
    inputs = p.add_mutually_exclusive_group()
    inputs.add_argument("--config", help="JSON network configuration")
    inputs.add_argument("--network", help="Saved network.npz input")
    p.add_argument("--method", choices=METHODS, default="joint")
    p.add_argument("--seconds", type=float, default=60)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", required=True)
    p.add_argument("--plot", action="store_true")
    p.set_defaults(handler=run)
    p = sub.add_parser("benchmark", help="Run independently budgeted methods on matched layouts")
    p.add_argument("--config")
    p.add_argument("--methods", default="joint,strong-cjt,pure-ncjt,fixed-hybrid,swap-switch")
    p.add_argument("--seconds", type=float, default=60)
    p.add_argument("--layouts", type=int, default=5)
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--fronthaul", help="Comma-separated mean capacities")
    p.add_argument("--max-aps", help="Comma-separated association limits")
    p.add_argument("--output", required=True)
    p.set_defaults(handler=benchmark)
    p = sub.add_parser("report", help="Regenerate a completed batch summary")
    p.add_argument("folder")
    p.set_defaults(handler=lambda args: summarize(args.folder))
    p = sub.add_parser("validate", help="Independently check saved network and solution arrays")
    p.add_argument("folder")
    p.set_defaults(handler=validate)
    args = parser.parse_args(argv)
    try:
        args.handler(args)
    except (ValueError, FileExistsError, FileNotFoundError, KeyError, TypeError) as error:
        parser.exit(2, f"cellfree: {error}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
