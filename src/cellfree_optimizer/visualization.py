"""Network and incumbent views for individual simulation runs."""

from pathlib import Path
import numpy as np


def plot_result(network, result, folder):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    )
    n, s = network.moments, result.solution
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.9), constrained_layout=True)
    ap, ue, z = network.ap_positions, network.user_positions, s["z"]
    colors = np.where(z, "#0d9488", "#ea580c")
    for m, k in zip(*np.nonzero(s["A"] & (s["x"] > 1e-6))):
        # A periodic link is drawn in the displayed square; distances use wrap-around.
        axes[0].plot([ap[m, 0], ue[k, 0]], [ap[m, 1], ue[k, 1]], color=colors[k], alpha=0.22, lw=1)
    axes[0].scatter(
        ap[:, 0], ap[:, 1], marker="^", s=65, c="#334155", label="Access points", zorder=3
    )
    for mode, color, label in [(True, "#0d9488", "CJT users"), (False, "#ea580c", "NCJT users")]:
        axes[0].scatter(
            ue[z == mode, 0], ue[z == mode, 1], marker="o", s=48, c=color, label=label, zorder=4
        )
    axes[0].set(
        xlim=(0, network.config.side_length),
        ylim=(0, network.config.side_length),
        xlabel="Horizontal position (m)",
        ylabel="Vertical position (m)",
        title="Serving links and modes",
    )
    axes[0].legend(loc="upper right", framealpha=0.94, fontsize=8)
    axes[0].grid(alpha=0.12)
    trace = result.metadata["trace"]
    times = [r["seconds"] for r in trace]
    service = [r["rate"] for r in trace]
    axes[1].step(
        times + [result.metadata["budget_seconds"]],
        service + [service[-1]],
        where="post",
        color="#0891b2",
        lw=2.2,
    )
    axes[1].set(
        xlim=(0, result.metadata["budget_seconds"]),
        ylim=(0, max(service[-1] * 1.12, 0.1)),
        xlabel="Elapsed optimizer time (s)",
        ylabel="Minimum service (bit/s/Hz)",
        title="Accepted feasible incumbents",
    )
    axes[1].grid(alpha=0.18)
    fig.suptitle(
        f"{result.method}  |  {n['M']} APs  |  {n['N']} antennas/AP  |  {n['K']} users",
        fontsize=13,
        weight="bold",
    )
    folder = Path(folder)
    for extension in ["svg", "png"]:
        fig.savefig(
            folder / f"overview.{extension}",
            dpi=170,
            metadata={"Creator": "cellfree-resource-optimizer"},
        )
    plt.close(fig)
