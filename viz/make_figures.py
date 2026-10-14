"""Generate article figures directly from the archived stress reports."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/stress-suite-v1.json"
SIGN_REPORT = ROOT / "reports/stress-v1/stress-groups-m5000-v1-signs.json"
OUTPUT = ROOT / "figures"
PARTITIONS = (
    ("groups-5-5", "5 + 5", "#287a78", "o"),
    ("groups-4-3-2-1", "4 + 3 + 2 + 1", "#d08b2e", "s"),
    ("groups-9-1", "9 + 1", "#a34255", "^"),
)
TRAIN = (50, 200, 1000, 5000)
OVERLAP = (0, 0.5, 0.9, 1)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelcolor": "#25323d", "text.color": "#25323d",
        "xtick.color": "#4d5b65", "ytick.color": "#4d5b65",
        "axes.edgecolor": "#a8b2b9", "axes.titleweight": "semibold",
        "svg.hashsalt": "shared-evidence-forecasting-v1",
    })


def _save(fig: plt.Figure, stem: str) -> None:
    for suffix in ("svg", "png"):
        path = OUTPUT / f"{stem}.{suffix}"
        fig.savefig(path, dpi=220,
                    bbox_inches="tight", facecolor="white", metadata={"Date": None})
        if suffix == "svg":
            # Matplotlib writes spaces at the ends of multi-line path data.
            # They are unnecessary in SVG and trip Git's whitespace check.
            content = path.read_text(encoding="utf-8")
            path.write_text("\n".join(line.rstrip() for line in content.splitlines()) + "\n",
                            encoding="utf-8")
    plt.close(fig)


def make_figures() -> dict[str, object]:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    signs = json.loads(SIGN_REPORT.read_text(encoding="utf-8"))
    if report["status"] != "descriptive_synthetic_stress_report":
        raise ValueError("unexpected stress report status")
    if report["stress_lock_sha256"] != _sha(ROOT / "configs/stress-suite-v1.lock.json"):
        raise ValueError("stress report does not match current suite lock")
    if signs["summary_sha256"] != _sha(ROOT / "reports/stress-v1/stress-groups-m5000-v1.json"):
        raise ValueError("sign diagnostic does not match archived summary")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    _style()

    fig, ax = plt.subplots(figsize=(8.8, 5.2), layout="constrained")
    for prefix, label, color, marker in PARTITIONS:
        means, lower, upper = [], [], []
        for size in TRAIN:
            phase = f"exploratory-stress-groups-m{size}-v1"
            cell = report["experiments"][phase]["cells"][f"{prefix}-rho1"]
            value = cell["contrasts"]["learned_minus_grouped_oracle"]["log_loss"]
            means.append(value["mean"])
            lower.append(value["mean"] - value["bootstrap_interval"][0])
            upper.append(value["bootstrap_interval"][1] - value["mean"])
        ax.errorbar(TRAIN, means, yerr=(lower, upper), label=label, color=color,
                    marker=marker, linewidth=2.1, capsize=3, markersize=6)
    ax.set_xscale("log")
    ax.set_xticks(TRAIN, [str(value) for value in TRAIN])
    ax.set_xlabel("Resolved training questions per fit")
    ax.set_ylabel("Learned minus group-oracle log loss\n(nats per test question)")
    ax.set_title("The source-structure gap persists with more training", loc="left", pad=15)
    ax.axhline(0, color="#7b858b", linewidth=1)
    ax.grid(axis="y", alpha=0.18)
    ax.legend(title="Independent source groups", frameon=False, loc="upper right")
    fig.text(0.01, -0.01, "Within-group overlap = 1. Error bars: descriptive 95% replicate bootstrap intervals; 12 fits per cell.",
             fontsize=9, color="#5b6871")
    _save(fig, "group_oracle_regret")

    fig, ax = plt.subplots(figsize=(8.8, 5.2), layout="constrained")
    for prefix, label, color, marker in PARTITIONS:
        values = [100 * signs["cells"][f"{prefix}-rho{rho:g}"]["opposite_sign_fraction"]
                  for rho in OVERLAP]
        ax.plot(OVERLAP, values, label=label, color=color, marker=marker,
                linewidth=2.1, markersize=6)
    ax.set_xticks(OVERLAP, [f"{value:g}" for value in OVERLAP])
    ax.set_xlabel("Within-group overlap")
    ax.set_ylabel("Opposite decision signs (%)")
    ax.set_title("Unequal source groups can reverse the pooled decision", loc="left", pad=15)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.18)
    ax.legend(title="Independent source groups", frameon=False, loc="upper left")
    fig.text(0.01, -0.01, "M = 5,000. Strict sign disagreement between summed reports and the group-aware oracle; 24,000 test questions per cell.",
             fontsize=9, color="#5b6871")
    _save(fig, "group_decision_disagreement")

    manifest = {
        "schema_version": 1,
        "status": "exploratory_figure_artifacts",
        "matplotlib_version": matplotlib.__version__,
        "input_sha256": {str(REPORT.relative_to(ROOT)): _sha(REPORT),
                         str(SIGN_REPORT.relative_to(ROOT)): _sha(SIGN_REPORT)},
        "script_sha256": _sha(Path(__file__)),
        "output_sha256": {path.name: _sha(path) for path in sorted(OUTPUT.glob("*.png")) + sorted(OUTPUT.glob("*.svg"))},
    }
    (OUTPUT / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    result = make_figures()
    print(f"Saved {len(result['output_sha256'])} figure files to {OUTPUT}")
