#!/usr/bin/env python3
"""Generate publication-quality charts and HTML dashboard from benchmark results."""

import json
import os
import base64
from pathlib import Path
from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

RESULTS_DIR = Path(__file__).parent.parent / "tracking_ws" / "data" / "results"
CHARTS_DIR = RESULTS_DIR / "charts"

MODELS = {
    "yolov8n_pretrained": "YOLOv8n pretrained",
    "yolov8n_finetuned": "YOLOv8n fine-tuned",
    "yolo26n_pretrained": "YOLO26n pretrained",
    "yolo26n_finetuned": "YOLO26n fine-tuned",
}

MODEL_COLORS = {
    "yolov8n_pretrained": "#58a6ff",
    "yolov8n_finetuned": "#1f6feb",
    "yolo26n_pretrained": "#f0883e",
    "yolo26n_finetuned": "#da3633",
}

ATTACK_ORDER = [
    "Baseline (clean)",
    "Fog (light)", "Fog (heavy)",
    "Rain (light)", "Rain (heavy)",
    "Blur (light)", "Blur (heavy)",
    "Low light", "Low light (extreme)",
    "Contrast loss",
    "Adv. patch", "Adv. stripe", "Checkerboard", "Occlusion",
    "FGSM (light)", "FGSM (heavy)",
    "PGD (light)", "PGD (heavy)",
]

ATTACK_CATEGORIES = {
    "Environmental": [
        "Fog (light)", "Fog (heavy)", "Rain (light)", "Rain (heavy)",
        "Blur (light)", "Blur (heavy)", "Low light", "Low light (extreme)",
        "Contrast loss",
    ],
    "Adversarial pattern": [
        "Adv. patch", "Adv. stripe", "Checkerboard", "Occlusion",
    ],
    "Gradient": [
        "FGSM (light)", "FGSM (heavy)", "PGD (light)", "PGD (heavy)",
    ],
}

# -- Dark theme for matplotlib --
DARK_BG = "#0d1117"
DARK_CARD = "#161b22"
DARK_BORDER = "#21262d"
TEXT_COLOR = "#c9d1d9"
MUTED_TEXT = "#8b949e"


def setup_dark_style():
    plt.rcParams.update({
        "figure.facecolor": DARK_BG,
        "axes.facecolor": DARK_CARD,
        "axes.edgecolor": DARK_BORDER,
        "axes.labelcolor": TEXT_COLOR,
        "text.color": TEXT_COLOR,
        "xtick.color": MUTED_TEXT,
        "ytick.color": MUTED_TEXT,
        "grid.color": DARK_BORDER,
        "grid.alpha": 0.6,
        "legend.facecolor": DARK_CARD,
        "legend.edgecolor": DARK_BORDER,
        "legend.labelcolor": TEXT_COLOR,
        "font.family": "sans-serif",
        "font.size": 11,
        "savefig.facecolor": DARK_BG,
        "savefig.edgecolor": DARK_BG,
    })


def load_results():
    """Load all results into {model_key: {attack_name: {MOTA, MOTP, ...}}}."""
    data = {}
    for model_key in MODELS:
        model_dir = RESULTS_DIR / model_key
        data[model_key] = {}
        for fname in ["results.json", "gradient.json"]:
            fpath = model_dir / fname
            if not fpath.exists():
                continue
            with open(fpath) as f:
                raw = json.load(f)
            for entry in raw.get("results", []):
                name = entry["name"]
                data[model_key][name] = entry
    return data


def get_mota(data, model, attack):
    entry = data.get(model, {}).get(attack)
    return entry["MOTA"] if entry else None


def chart_baseline_comparison(data):
    """Bar chart: 4 models, baseline MOTA side by side."""
    fig, ax = plt.subplots(figsize=(8, 5))
    models = list(MODELS.keys())
    labels = [MODELS[m] for m in models]
    values = [get_mota(data, m, "Baseline (clean)") or 0 for m in models]
    colors = [MODEL_COLORS[m] for m in models]

    bars = ax.bar(range(len(models)), values, color=colors, width=0.6, edgecolor=DARK_BORDER)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("MOTA")
    ax.set_title("Baseline Performance (Clean, No Attacks)", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylim(0, max(values) * 1.25)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold", color=TEXT_COLOR)

    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "baseline_comparison.png", dpi=150)
    plt.close(fig)


def chart_mota_all_attacks(data):
    """Grouped bar chart: all models x all 18 attacks."""
    attacks = ATTACK_ORDER[1:]  # skip baseline
    models = list(MODELS.keys())
    n_attacks = len(attacks)
    n_models = len(models)
    x = np.arange(n_attacks)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(18, 7))

    for i, model in enumerate(models):
        values = []
        for attack in attacks:
            v = get_mota(data, model, attack)
            values.append(v if v is not None else float("nan"))
        offset = (i - (n_models - 1) / 2) * width
        bars = ax.bar(x + offset, values, width, label=MODELS[model],
                      color=MODEL_COLORS[model], edgecolor=DARK_BORDER, linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(attacks, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("MOTA")
    ax.set_title("MOTA Under Attack — All Models", fontsize=14, fontweight="bold", pad=12)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.axhline(y=0, color=MUTED_TEXT, linewidth=0.8, linestyle="-")

    # Category separators
    env_end = len(ATTACK_CATEGORIES["Environmental"]) - 0.5
    adv_end = env_end + len(ATTACK_CATEGORIES["Adversarial pattern"])
    for sep in [env_end, adv_end]:
        ax.axvline(x=sep, color=MUTED_TEXT, linewidth=0.8, linestyle=":", alpha=0.5)

    # Category labels at top
    ax.text(env_end / 2, ax.get_ylim()[1] * 0.95, "Environmental",
            ha="center", fontsize=9, color=MUTED_TEXT, style="italic")
    ax.text((env_end + adv_end) / 2, ax.get_ylim()[1] * 0.95, "Adversarial",
            ha="center", fontsize=9, color=MUTED_TEXT, style="italic")
    ax.text((adv_end + n_attacks - 1) / 2, ax.get_ylim()[1] * 0.95, "Gradient",
            ha="center", fontsize=9, color=MUTED_TEXT, style="italic")

    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "mota_all_attacks.png", dpi=150)
    plt.close(fig)


def chart_degradation_heatmap(data):
    """Heatmap: attacks as rows, models as columns, color = % MOTA drop from baseline."""
    attacks = ATTACK_ORDER[1:]
    models = list(MODELS.keys())

    matrix = []
    for attack in attacks:
        row = []
        for model in models:
            baseline = get_mota(data, model, "Baseline (clean)")
            val = get_mota(data, model, attack)
            if baseline is not None and val is not None and baseline != 0:
                pct_drop = ((val - baseline) / abs(baseline)) * 100
            else:
                pct_drop = float("nan")
            row.append(pct_drop)
        matrix.append(row)

    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(10, 12))
    cmap = mcolors.LinearSegmentedColormap.from_list("rg", ["#f85149", "#d29922", "#3fb950"])
    norm = mcolors.TwoSlopeNorm(vmin=-120, vcenter=-50, vmax=10)

    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([MODELS[m] for m in models], rotation=20, ha="right", fontsize=10)
    ax.set_yticks(range(len(attacks)))
    ax.set_yticklabels(attacks, fontsize=10)
    ax.set_title("MOTA Degradation from Baseline (%)", fontsize=14, fontweight="bold", pad=12)

    # Annotate cells
    for i in range(len(attacks)):
        for j in range(len(models)):
            val = matrix[i, j]
            if np.isnan(val):
                text = "—"
                color = MUTED_TEXT
            else:
                text = f"{val:+.1f}%"
                color = "#ffffff" if val < -60 else TEXT_COLOR
            ax.text(j, i, text, ha="center", va="center", fontsize=9, color=color, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("% Change from Baseline", color=MUTED_TEXT)
    cbar.ax.yaxis.set_tick_params(color=MUTED_TEXT)
    for label in cbar.ax.get_yticklabels():
        label.set_color(MUTED_TEXT)

    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "degradation_heatmap.png", dpi=150)
    plt.close(fig)


def chart_attack_categories(data):
    """Bar chart grouped by attack category, showing avg MOTA drop per model."""
    models = list(MODELS.keys())
    categories = list(ATTACK_CATEGORIES.keys())
    n_cats = len(categories)
    n_models = len(models)
    x = np.arange(n_cats)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, model in enumerate(models):
        baseline = get_mota(data, model, "Baseline (clean)")
        avg_drops = []
        for cat in categories:
            drops = []
            for attack in ATTACK_CATEGORIES[cat]:
                val = get_mota(data, model, attack)
                if val is not None and baseline is not None and baseline != 0:
                    drops.append(((val - baseline) / abs(baseline)) * 100)
            avg_drops.append(np.mean(drops) if drops else float("nan"))

        offset = (i - (n_models - 1) / 2) * width
        bars = ax.bar(x + offset, avg_drops, width, label=MODELS[model],
                      color=MODEL_COLORS[model], edgecolor=DARK_BORDER, linewidth=0.5)

        for bar, val in zip(bars, avg_drops):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 2,
                        f"{val:.0f}%", ha="center", va="top", fontsize=8,
                        fontweight="bold", color="#ffffff")

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylabel("Avg. MOTA Change from Baseline (%)")
    ax.set_title("Impact by Attack Category", fontsize=14, fontweight="bold", pad=12)
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.axhline(y=0, color=MUTED_TEXT, linewidth=0.8, linestyle="-")

    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "attack_categories.png", dpi=150)
    plt.close(fig)


def chart_gradient_robustness(data):
    """Focused comparison of v8 vs v26 under gradient attacks (the key finding)."""
    gradient_attacks = ["FGSM (light)", "FGSM (heavy)", "PGD (light)", "PGD (heavy)"]
    # Compare finetuned models (the interesting comparison)
    models = ["yolov8n_finetuned", "yolo26n_finetuned"]
    x = np.arange(len(gradient_attacks))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: absolute MOTA
    for i, model in enumerate(models):
        values = [get_mota(data, model, a) or 0 for a in gradient_attacks]
        offset = (i - 0.5) * width
        bars = ax1.bar(x + offset, values, width, label=MODELS[model],
                       color=MODEL_COLORS[model], edgecolor=DARK_BORDER)
        for bar, val in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=9, color=TEXT_COLOR)

    # Add baseline lines
    for model in models:
        bl = get_mota(data, model, "Baseline (clean)")
        if bl is not None:
            ax1.axhline(y=bl, color=MODEL_COLORS[model], linewidth=1.2,
                        linestyle="--", alpha=0.6, label=f"{MODELS[model]} baseline")

    ax1.set_xticks(x)
    ax1.set_xticklabels(gradient_attacks, rotation=15, ha="right", fontsize=10)
    ax1.set_ylabel("MOTA")
    ax1.set_title("Absolute MOTA Under Gradient Attacks", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    # Right: % drop from baseline
    for i, model in enumerate(models):
        baseline = get_mota(data, model, "Baseline (clean)")
        drops = []
        for a in gradient_attacks:
            val = get_mota(data, model, a)
            if val is not None and baseline is not None and baseline != 0:
                drops.append(((val - baseline) / abs(baseline)) * 100)
            else:
                drops.append(0)
        offset = (i - 0.5) * width
        bars = ax2.bar(x + offset, drops, width, label=MODELS[model],
                       color=MODEL_COLORS[model], edgecolor=DARK_BORDER)
        for bar, val in zip(bars, drops):
            y = bar.get_height()
            va = "top" if y < 0 else "bottom"
            ax2.text(bar.get_x() + bar.get_width() / 2, y + (-1 if y < 0 else 1),
                     f"{val:.1f}%", ha="center", va=va, fontsize=9,
                     fontweight="bold", color=TEXT_COLOR)

    ax2.set_xticks(x)
    ax2.set_xticklabels(gradient_attacks, rotation=15, ha="right", fontsize=10)
    ax2.set_ylabel("MOTA Change from Baseline (%)")
    ax2.set_title("Relative Degradation (%)", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=8, loc="lower left")
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    ax2.axhline(y=0, color=MUTED_TEXT, linewidth=0.8, linestyle="-")

    fig.suptitle("Gradient Attack Robustness: YOLOv8 vs YOLO26 (Fine-Tuned)",
                 fontsize=14, fontweight="bold", y=1.02, color=TEXT_COLOR)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "gradient_robustness.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def png_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def generate_dashboard(data):
    """Generate self-contained HTML dashboard."""
    charts = {}
    for name in ["baseline_comparison", "mota_all_attacks", "degradation_heatmap",
                  "attack_categories", "gradient_robustness"]:
        p = CHARTS_DIR / f"{name}.png"
        if p.exists():
            charts[name] = png_to_base64(p)

    # Build results table rows
    table_rows = ""
    for attack in ATTACK_ORDER:
        cells = f'<td class="attack-name">{attack}</td>'
        for model in MODELS:
            val = get_mota(data, model, attack)
            if val is None:
                cells += '<td class="na">—</td>'
            elif attack == "Baseline (clean)":
                cells += f'<td class="baseline-val">{val:.3f}</td>'
            elif val < 0:
                cells += f'<td class="negative">{val:.3f}</td>'
            else:
                cells += f'<td>{val:.3f}</td>'
        # Degradation columns (finetuned models only)
        for model in ["yolov8n_finetuned", "yolo26n_finetuned"]:
            baseline = get_mota(data, model, "Baseline (clean)")
            val = get_mota(data, model, attack)
            if attack == "Baseline (clean)":
                cells += '<td class="baseline-val">—</td>'
            elif val is not None and baseline is not None and baseline != 0:
                pct = ((val - baseline) / abs(baseline)) * 100
                cls = "drop-mild" if pct > -25 else "drop-moderate" if pct > -60 else "drop-severe"
                cells += f'<td class="{cls}">{pct:+.1f}%</td>'
            else:
                cells += '<td class="na">—</td>'

        row_cls = "baseline-row" if attack == "Baseline (clean)" else ""
        table_rows += f"<tr class=\"{row_cls}\">{cells}</tr>\n"

    # Key metrics
    metrics = {}
    for model in MODELS:
        bl = get_mota(data, model, "Baseline (clean)")
        metrics[model] = {"baseline": bl}
        worst_name, worst_val = None, 999
        for attack in ATTACK_ORDER[1:]:
            v = get_mota(data, model, attack)
            if v is not None and v < worst_val:
                worst_val = v
                worst_name = attack
        metrics[model]["worst"] = (worst_name, worst_val)

    # Gradient robustness summary
    v8_pgd_heavy = get_mota(data, "yolov8n_finetuned", "PGD (heavy)")
    v8_bl = get_mota(data, "yolov8n_finetuned", "Baseline (clean)")
    v26_pgd_heavy = get_mota(data, "yolo26n_finetuned", "PGD (heavy)")
    v26_bl = get_mota(data, "yolo26n_finetuned", "Baseline (clean)")

    v8_pgd_drop = ((v8_pgd_heavy - v8_bl) / abs(v8_bl)) * 100 if v8_bl and v8_pgd_heavy else 0
    v26_pgd_drop = ((v26_pgd_heavy - v26_bl) / abs(v26_bl)) * 100 if v26_bl and v26_pgd_heavy else 0

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Adversarial Tracking Dashboard</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ background: {DARK_BG}; color: {TEXT_COLOR}; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif; padding: 40px 20px; max-width: 1400px; margin: 0 auto; }}
h1 {{ color: #58a6ff; font-size: 28px; margin-bottom: 8px; }}
h2 {{ color: #58a6ff; font-size: 20px; margin: 40px 0 16px; border-bottom: 1px solid {DARK_BORDER}; padding-bottom: 8px; }}
.subtitle {{ color: {MUTED_TEXT}; margin-bottom: 30px; font-size: 15px; }}
.metrics-row {{ display: flex; gap: 16px; margin-bottom: 30px; flex-wrap: wrap; }}
.metric-card {{ background: {DARK_CARD}; border: 1px solid {DARK_BORDER}; border-radius: 8px; padding: 20px; flex: 1; min-width: 200px; }}
.metric-card .value {{ font-size: 28px; font-weight: bold; margin-bottom: 4px; }}
.metric-card .label {{ color: {MUTED_TEXT}; font-size: 13px; }}
.metric-card .sublabel {{ color: {MUTED_TEXT}; font-size: 11px; margin-top: 4px; }}
.good .value {{ color: #3fb950; }}
.warn .value {{ color: #d29922; }}
.bad .value {{ color: #f85149; }}
.info .value {{ color: #58a6ff; }}
.chart-section {{ background: {DARK_CARD}; border: 1px solid {DARK_BORDER}; border-radius: 8px; padding: 24px; margin-bottom: 24px; }}
.chart-section img {{ width: 100%; border-radius: 4px; }}
.chart-caption {{ color: {MUTED_TEXT}; font-size: 12px; margin-top: 8px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; margin-bottom: 24px; }}
th {{ background: {DARK_CARD}; color: {MUTED_TEXT}; padding: 10px 8px; text-align: right; font-weight: 600; border-bottom: 2px solid {DARK_BORDER}; position: sticky; top: 0; }}
th:first-child {{ text-align: left; }}
td {{ padding: 8px; border-bottom: 1px solid {DARK_BORDER}; text-align: right; font-variant-numeric: tabular-nums; }}
td.attack-name {{ text-align: left; font-weight: 500; }}
td.na {{ color: {MUTED_TEXT}; }}
td.negative {{ color: #f85149; }}
td.baseline-val {{ color: #3fb950; font-weight: 600; }}
td.drop-mild {{ color: #3fb950; }}
td.drop-moderate {{ color: #d29922; }}
td.drop-severe {{ color: #f85149; font-weight: 600; }}
tr.baseline-row {{ background: rgba(88, 166, 255, 0.06); }}
tr:hover {{ background: rgba(255, 255, 255, 0.03); }}
.findings {{ background: {DARK_CARD}; border: 1px solid {DARK_BORDER}; border-radius: 8px; padding: 24px; }}
.findings ol {{ padding-left: 20px; }}
.findings li {{ margin-bottom: 12px; line-height: 1.5; }}
.findings strong {{ color: #58a6ff; }}
.highlight {{ background: rgba(218, 54, 51, 0.15); border-left: 3px solid #da3633; padding: 16px; border-radius: 0 8px 8px 0; margin: 16px 0; }}
.highlight strong {{ color: #f85149; }}
.footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid {DARK_BORDER}; color: #484f58; font-size: 12px; }}
.two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
@media (max-width: 900px) {{ .two-col {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>

<h1>Adversarial Tracking Benchmark Dashboard</h1>
<p class="subtitle">4 models &times; 18 attack configurations &mdash; VisDrone2019-MOT validation set</p>

<div class="metrics-row">
<div class="metric-card good">
    <div class="value">{metrics['yolov8n_finetuned']['baseline']:.3f}</div>
    <div class="label">Best Baseline MOTA</div>
    <div class="sublabel">YOLOv8n fine-tuned</div>
</div>
<div class="metric-card info">
    <div class="value">{v26_pgd_drop:+.1f}%</div>
    <div class="label">YOLO26 FT: PGD Heavy Drop</div>
    <div class="sublabel">vs {v8_pgd_drop:+.1f}% for YOLOv8 FT</div>
</div>
<div class="metric-card bad">
    <div class="value">{metrics['yolov8n_finetuned']['worst'][1]:.3f}</div>
    <div class="label">Worst MOTA (v8 FT)</div>
    <div class="sublabel">{metrics['yolov8n_finetuned']['worst'][0]}</div>
</div>
<div class="metric-card warn">
    <div class="value">18</div>
    <div class="label">Attack Configurations</div>
    <div class="sublabel">3 categories tested</div>
</div>
</div>

<h2>Baseline Performance</h2>
<div class="chart-section">
    <img src="data:image/png;base64,{charts.get('baseline_comparison', '')}" alt="Baseline comparison">
    <p class="chart-caption">Clean performance without any attacks. Fine-tuning on VisDrone roughly triples MOTA for both architectures.</p>
</div>

<h2>MOTA Under All Attacks</h2>
<div class="chart-section">
    <img src="data:image/png;base64,{charts.get('mota_all_attacks', '')}" alt="MOTA all attacks">
    <p class="chart-caption">All 17 attack configurations across 4 models. Vertical dotted lines separate attack categories.</p>
</div>

<div class="two-col">
<div>
<h2>Degradation Heatmap</h2>
<div class="chart-section">
    <img src="data:image/png;base64,{charts.get('degradation_heatmap', '')}" alt="Degradation heatmap">
    <p class="chart-caption">Percentage MOTA change from each model's baseline. Green = minimal impact, red = severe degradation.</p>
</div>
</div>
<div>
<h2>Impact by Category</h2>
<div class="chart-section">
    <img src="data:image/png;base64,{charts.get('attack_categories', '')}" alt="Attack categories">
    <p class="chart-caption">Average MOTA degradation per attack category. Adversarial pattern attacks are universally devastating.</p>
</div>
</div>
</div>

<h2>Gradient Attack Robustness (Key Finding)</h2>
<div class="chart-section">
    <img src="data:image/png;base64,{charts.get('gradient_robustness', '')}" alt="Gradient robustness">
    <p class="chart-caption">Fine-tuned models compared under white-box gradient attacks. YOLO26 shows dramatically less degradation despite lower baseline.</p>
</div>

<div class="highlight">
    <strong>Key result:</strong> PGD heavy drops YOLOv8n fine-tuned by {abs(v8_pgd_drop):.1f}% but YOLO26n fine-tuned by only {abs(v26_pgd_drop):.1f}%.
    The attention-based architecture appears to diffuse adversarial gradients, making white-box attacks significantly less effective.
</div>

<h2>Full Results Table</h2>
<div style="overflow-x: auto;">
<table>
<thead>
<tr>
    <th style="text-align:left">Attack</th>
    <th>v8 pretrained</th>
    <th>v8 fine-tuned</th>
    <th>v26 pretrained</th>
    <th>v26 fine-tuned</th>
    <th>v8 FT drop</th>
    <th>v26 FT drop</th>
</tr>
</thead>
<tbody>
{table_rows}
</tbody>
</table>
</div>

<h2>Key Findings</h2>
<div class="findings">
<ol>
<li><strong>Fine-tuning is essential.</strong> Both models roughly tripled baseline MOTA after fine-tuning on VisDrone. Domain adaptation matters more than architecture choice for absolute tracking performance.</li>
<li><strong>YOLOv8n outperforms YOLO26n at nano scale.</strong> Despite being a newer architecture with attention mechanisms, YOLO26n&rsquo;s design overhead hurts at the smallest model size ({metrics['yolov8n_finetuned']['baseline']:.3f} vs {metrics['yolo26n_finetuned']['baseline']:.3f} fine-tuned baseline).</li>
<li><strong>YOLO26 fine-tuned is remarkably robust to gradient attacks.</strong> PGD heavy drops YOLOv8 FT by {abs(v8_pgd_drop):.1f}% but YOLO26 FT by only {abs(v26_pgd_drop):.1f}%. The attention-based architecture appears to diffuse adversarial gradients.</li>
<li><strong>Adversarial pattern attacks are devastating regardless of architecture.</strong> Stripe, checkerboard, and patch attacks drive MOTA to zero or negative across all models.</li>
<li><strong>Environmental attacks show a robustness-accuracy tradeoff.</strong> YOLO26 FT shows smaller relative drops under rain and blur despite lower absolute MOTA.</li>
<li><strong>Low light (extreme) is a universal failure mode.</strong> All models collapse to MOTA &asymp; 0.000 &mdash; a fundamental limit of visual perception.</li>
</ol>
</div>

<div class="footer">
    Generated by <code>scripts/visualize_results.py</code> &mdash; Adversarial Tracking Benchmark
</div>

</body>
</html>"""

    with open(RESULTS_DIR / "dashboard.html", "w") as f:
        f.write(html)


def main():
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    setup_dark_style()

    print("Loading results...")
    data = load_results()
    for model in MODELS:
        n = len(data.get(model, {}))
        print(f"  {MODELS[model]}: {n} configs")

    print("\nGenerating charts...")
    chart_baseline_comparison(data)
    print("  [1/5] baseline_comparison.png")

    chart_mota_all_attacks(data)
    print("  [2/5] mota_all_attacks.png")

    chart_degradation_heatmap(data)
    print("  [3/5] degradation_heatmap.png")

    chart_attack_categories(data)
    print("  [4/5] attack_categories.png")

    chart_gradient_robustness(data)
    print("  [5/5] gradient_robustness.png")

    print("\nGenerating dashboard...")
    generate_dashboard(data)
    print(f"  dashboard.html")

    print(f"\nDone! Output in {CHARTS_DIR.relative_to(Path.cwd())}/ and {(RESULTS_DIR / 'dashboard.html').relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
