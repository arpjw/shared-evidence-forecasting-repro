"""Plot the frozen source-inference result from its verified report."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/source-inference-suite-v1.json"
LOCK = ROOT / "configs/source-inference-suite-v1.lock.json"
OUTPUT = ROOT / "figures"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(fig: plt.Figure, stem: str) -> None:
    for suffix in ("png", "svg"):
        path = OUTPUT / f"{stem}.{suffix}"
        fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white", metadata={"Date": None})
        if suffix == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)


def main() -> None:
    report = json.loads(REPORT.read_text())
    if (report["status"] != "independently_verified_locked_synthetic_evaluation"
            or report["lock_sha256"] != digest(LOCK)):
        raise ValueError("figure input is not the verified frozen suite")
    primary = report["phases"]["source-inference-primary-v1"]["conditions"]["groups-9-1-rho0.9"]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
        "axes.spines.right": False, "axes.labelcolor": "#25323d",
        "text.color": "#25323d", "xtick.color": "#4d5b65",
        "ytick.color": "#4d5b65", "axes.edgecolor": "#a8b2b9",
        "svg.hashsalt": "shared-evidence-source-inference-v1",
    })
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.9), layout="constrained")
    methods = (("scalar", "Single scalar", "#a34255"),
               ("inferred_groups", "Inferred sources", "#287a78"),
               ("grouped_oracle", "Known sources", "#506d96"))
    for y, (key, label, color) in zip((2, 1, 0), methods):
        value = primary["methods"][key]["log_loss"]["mean"]
        ax1.scatter(value, y, color=color, s=80, zorder=3)
        ax1.text(value + 0.002, y, f"{value:.5f}", va="center", fontsize=10)
    ax1.set_yticks((2, 1, 0), [item[1] for item in methods])
    ax1.set_xlim(0.47, 0.57)
    ax1.set_ylim(-0.6, 2.6)
    ax1.set_xlabel("Mean held-out log loss (nats)")
    ax1.set_title("A. Fresh 9 + 1 test, overlap .9", loc="left", pad=12, fontweight="semibold")
    ax1.grid(axis="x", alpha=0.18)

    series = (("groups-9-1-rho0.9", "9 + 1, overlap .9", "#a34255", "o"),
              ("groups-4-3-2-1-rho0.5", "4 + 3 + 2 + 1, overlap .5", "#d08b2e", "s"))
    for name, label, color, marker in series:
        x_values, means, lower, upper = [], [], [], []
        for m in (100, 400, 2000):
            phase = report["phases"][f"source-inference-sensitivity-m{m}-v1"]
            contrast = phase["conditions"][name]["contrasts"]["inferred_groups_minus_grouped_oracle"]["log_loss"]
            x_values.append(m)
            means.append(contrast["mean"])
            lower.append(contrast["mean"] - contrast["bootstrap_interval"][0])
            upper.append(contrast["bootstrap_interval"][1] - contrast["mean"])
        ax2.errorbar(x_values, means, yerr=(lower, upper), label=label, color=color,
                     marker=marker, linewidth=2, capsize=3, markersize=6)
    ax2.axhline(0, color="#7b858b", linewidth=1)
    ax2.set_xscale("log")
    ax2.set_xticks((100, 400, 2000), ("100", "400", "2,000"))
    ax2.set_xlabel("Resolved training questions per fit")
    ax2.set_ylabel("Inferred minus oracle log loss (nats)")
    ax2.set_title("B. Remaining regret with more training", loc="left", pad=12, fontweight="semibold")
    ax2.grid(axis="y", alpha=0.18)
    ax2.legend(frameon=False, loc="upper right", fontsize=9)
    fig.text(0.01, -0.04, "Panel A: 30 independent fits and 150,000 test questions. Panel B: 12 fits per cell; descriptive 95% replicate-bootstrap intervals.",
             fontsize=9, color="#5b6871")
    save(fig, "source_inference_regret")
    manifest = {
        "schema_version": 1, "status": "verified_source_inference_figure",
        "matplotlib_version": matplotlib.__version__,
        "input_sha256": {str(REPORT.relative_to(ROOT)): digest(REPORT),
                         str(LOCK.relative_to(ROOT)): digest(LOCK)},
        "script_sha256": digest(Path(__file__)),
        "output_sha256": {path.name: digest(path) for path in
                          (OUTPUT / "source_inference_regret.png", OUTPUT / "source_inference_regret.svg")},
    }
    (OUTPUT / "source_inference_figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
