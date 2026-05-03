#!/usr/bin/env python3
"""
Stability analysis of resume scoring by demographic group, broken down per model.

Reads experiments/raw_scores.jsonl and computes two complementary metrics:

  Score Stability  — 1 − (stdev / MAX_SD) per resume, then averaged by group.
                     Captures absolute score consistency across paraphrases.

  Rank Stability   — Average pairwise Spearman ρ across paraphrase versions,
                     computed at the group level. Captures whether the relative
                     ordering of resumes is preserved across rephrasings.

Outputs (written to this analysis/ directory):
  race_stability_by_model[_<cat>].csv / .json
  gender_stability_by_model[_<cat>].csv / .json
  race_rank_stability_by_model[_<cat>].csv / .json
  gender_rank_stability_by_model[_<cat>].csv / .json
  plots/stability_heatmap_race[_<cat>].png
  plots/stability_heatmap_gender[_<cat>].png
  plots/score_dist_by_race[_<cat>].png
  plots/score_dist_by_gender[_<cat>].png
"""

import argparse
import json
import csv
import os
from itertools import combinations
from statistics import stdev, mean
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR  = os.path.dirname(ANALYSIS_DIR)
RAW_SCORES   = os.path.join(PROJECT_DIR, "experiments", "raw_scores_new_prompt.jsonl")
PLOTS_DIR    = os.path.join(ANALYSIS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

MAX_SD = 4.5  # worst-case stdev for 1–10 scale: stdev([1]*5 + [10]*5)

RACE_ORDER   = ["White", "Black or African American", "Asian", "Hispanic or Latino"]
RACE_SHORT   = {
    "White":                      "White",
    "Black or African American":  "Black",
    "Asian":                      "Asian",
    "Hispanic or Latino":         "Hispanic",
}
GENDER_ORDER = ["man", "woman"]
CATEGORY_LABELS = {
    "INFORMATION-TECHNOLOGY": "IT",
    "DESIGNER":               "Designer",
}

# ---------------------------------------------------------------------------
# Core helpers — score stability
# ---------------------------------------------------------------------------

def compute_stability(scores: list) -> float:
    if len(scores) < 2:
        return 1.0
    sd = stdev(scores)
    return round(max(0.0, 1.0 - sd / MAX_SD), 4)


def load_raw_scores(path: str) -> list:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def per_resume_stability(rows: list) -> list:
    """Score stability for every (model, resume_id, category, race, gender) group."""
    groups = defaultdict(list)
    for r in rows:
        key = (r["model_short"], r["resume_id"], r["category"], r["race"], r["gender"])
        groups[key].append(r["score"])

    records = []
    for (model, resume_id, category, race, gender), scores in groups.items():
        records.append({
            "model_short": model,
            "resume_id":   resume_id,
            "category":    category,
            "race":        race,
            "gender":      gender,
            "stability":   compute_stability(scores),
            "n_versions":  len(scores),
        })
    return records


def aggregate_by_model(stability_records: list, group_key: str) -> list:
    """Average score stability by (model_short, category, group_key)."""
    buckets = defaultdict(list)
    for r in stability_records:
        buckets[(r["model_short"], r["category"], r[group_key])].append(r["stability"])

    rows = []
    for (model, category, group_val), stabs in sorted(buckets.items()):
        rows.append({
            "model_short":   model,
            "category":      category,
            group_key:       group_val,
            "avg_stability": round(mean(stabs), 4),
            "count":         len(stabs),
        })
    return rows


# ---------------------------------------------------------------------------
# Core helpers — rank stability
# ---------------------------------------------------------------------------

def compute_rank_stability_by_group(rows: list, group_key: str) -> list:
    """
    For each (model, category, group_value), compute average pairwise Spearman ρ
    of resume score rankings across paraphrase versions.

    This answers: does the model rank the same resumes consistently regardless
    of which paraphrase it sees?
    """
    # nested[model][category][group_val][version] = {resume_id: score}
    nested: dict = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(dict)
            )
        )
    )
    for r in rows:
        nested[r["model_short"]][r["category"]][r[group_key]][r["version"]][r["resume_id"]] = r["score"]

    records = []
    for model, cats in nested.items():
        for category, groups in cats.items():
            for group_val, versions in groups.items():
                version_list = sorted(versions.keys())
                if len(version_list) < 2:
                    continue

                # Only use resumes that appear in every version
                common = set.intersection(*(set(versions[v].keys()) for v in version_list))
                if len(common) < 3:
                    continue
                common = sorted(common)

                corrs = []
                for v1, v2 in combinations(version_list, 2):
                    x = [versions[v1][rid] for rid in common]
                    y = [versions[v2][rid] for rid in common]
                    if np.std(x) > 0 and np.std(y) > 0:
                        rho, _ = spearmanr(x, y)
                        if not np.isnan(rho):
                            corrs.append(rho)

                if corrs:
                    records.append({
                        "model_short":        model,
                        "category":           category,
                        group_key:            group_val,
                        "avg_rank_stability": round(mean(corrs), 4),
                        "n_resumes":          len(common),
                        "n_version_pairs":    len(corrs),
                    })
    return records


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save_csv(data: list, path: str):
    if not data:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0].keys()))
        writer.writeheader()
        writer.writerows(data)
    print(f"  Saved {os.path.relpath(path, PROJECT_DIR)}")


def save_json(data: list, path: str):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved {os.path.relpath(path, PROJECT_DIR)}")


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _ordered_groups(data: list, group_key: str, preferred_order: list) -> list:
    present = {r[group_key] for r in data}
    ordered = [g for g in preferred_order if g in present]
    ordered += sorted(present - set(ordered))
    return ordered


def _draw_heatmap_row(ax, matrix, models, groups, cmap, vmin, vmax, show_xticks):
    """Draw a single heatmap into ax and return the image object."""
    n_groups = len(groups)
    n_models = len(models)

    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    for i in range(n_groups):
        for j in range(n_models):
            val = matrix[i, j]
            if not np.isnan(val):
                # Pick readable text colour based on background darkness
                norm_val = (val - vmin) / (vmax - vmin)
                text_color = "white" if norm_val > 0.65 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=8, color=text_color)

    ax.set_xticks(range(n_models))
    if show_xticks:
        ax.set_xticklabels(models, rotation=40, ha="right", fontsize=9)
    else:
        ax.set_xticklabels([])
    ax.set_yticks(range(n_groups))
    ax.set_yticklabels(groups, fontsize=10)
    return im


# ---------------------------------------------------------------------------
# Plot 1 — stability heatmaps (score + rank) per model
# ---------------------------------------------------------------------------

def plot_stability_heatmap(
    score_data: list,
    rank_data: list,
    group_key: str,
    title: str,
    out_path: str,
):
    """
    Two-row heatmap per job category:
      Row 0 — Score Stability  (1 − σ/σ_max), RdYlGn colormap
      Row 1 — Rank Stability   (avg Spearman ρ), Blues colormap
    """
    order      = RACE_ORDER if group_key == "race" else GENDER_ORDER
    categories = [c for c in CATEGORY_LABELS if any(r["category"] == c for r in score_data)]
    groups     = _ordered_groups(score_data, group_key, order)
    models     = sorted(set(r["model_short"] for r in score_data))

    n_cats   = len(categories)
    n_groups = len(groups)
    n_models = len(models)

    row_height = max(3, n_groups * 0.9) + 1.5
    fig, axes = plt.subplots(
        2, n_cats,
        figsize=(max(8, n_models * 1.1) * n_cats, row_height * 2),
        squeeze=False,
    )

    score_lookup = {
        (r["model_short"], r["category"], r[group_key]): r["avg_stability"]
        for r in score_data
    }
    rank_lookup = {
        (r["model_short"], r["category"], r[group_key]): r["avg_rank_stability"]
        for r in rank_data
    }

    metrics = [
        ("Score Stability  (1 − σ/σ_max)", score_lookup, "RdYlGn", 0.70, 1.00),
        ("Rank Stability  (Spearman ρ)",    rank_lookup,  "RdYlGn", 0.00, 1.00),
    ]

    for row, (row_label, lookup, cmap, vmin, vmax) in enumerate(metrics):
        show_xticks = (row == 1)  # x-tick labels only on bottom row
        for col, cat in enumerate(categories):
            ax = axes[row][col]
            matrix = np.array([
                [lookup.get((m, cat, g), np.nan) for m in models]
                for g in groups
            ])

            im = _draw_heatmap_row(ax, matrix, models, groups, cmap, vmin, vmax, show_xticks)

            if row == 0:
                ax.set_title(CATEGORY_LABELS.get(cat, cat), fontsize=12, fontweight="bold",
                             pad=6)

            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label(row_label, fontsize=8)

        # Row label on the y-axis of the first column
        axes[row][0].set_ylabel(row_label, fontsize=9, labelpad=8)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {os.path.relpath(out_path, PROJECT_DIR)}")


# ---------------------------------------------------------------------------
# Plot 2 — score distributions per model
# ---------------------------------------------------------------------------

def plot_score_distribution_by_model(rows: list, group_key: str, title: str, out_path: str):
    """
    Grid of violin plots: rows = job categories, cols = models.
    Each cell shows the score distribution broken down by demographic group.
    """
    order      = RACE_ORDER if group_key == "race" else GENDER_ORDER
    categories = [c for c in CATEGORY_LABELS if any(r["category"] == c for r in rows)]
    models     = sorted(set(r["model_short"] for r in rows))
    groups     = _ordered_groups(rows, group_key, order)

    short_labels = [RACE_SHORT.get(g, g) if group_key == "race" else g for g in groups]

    n_cats   = len(categories)
    n_models = len(models)
    cat_colors = ["#4C72B0", "#DD8452"]

    fig, axes = plt.subplots(
        n_cats, n_models,
        figsize=(n_models * 2.2, n_cats * 3.5),
        sharey=True, squeeze=False,
    )

    for row_i, cat in enumerate(categories):
        for col_j, model in enumerate(models):
            ax = axes[row_i][col_j]
            data_by_group = []
            labels        = []
            for g, lbl in zip(groups, short_labels):
                scores = [r["score"] for r in rows
                          if r["category"] == cat
                          and r["model_short"] == model
                          and r[group_key] == g]
                if scores:
                    data_by_group.append(scores)
                    labels.append(lbl)

            if data_by_group:
                positions = list(range(len(labels)))
                parts = ax.violinplot(data_by_group, positions=positions,
                                      showmedians=True, showextrema=False)
                for pc in parts["bodies"]:
                    pc.set_facecolor(cat_colors[row_i % len(cat_colors)])
                    pc.set_alpha(0.55)
                parts["cmedians"].set_color("black")
                parts["cmedians"].set_linewidth(1.4)

                for pos, scores in zip(positions, data_by_group):
                    ax.scatter(pos, mean(scores), color="white", edgecolors="black",
                               zorder=3, s=18, linewidths=0.8)

            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
            ax.set_ylim(0, 11)
            ax.set_yticks(range(1, 11, 2))
            ax.tick_params(axis="y", labelsize=7)
            ax.grid(axis="y", alpha=0.3)

            if col_j == 0:
                ax.set_ylabel(f"{CATEGORY_LABELS.get(cat, cat)}\nScore (1–10)", fontsize=9)
            if row_i == 0:
                ax.set_title(model, fontsize=8)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {os.path.relpath(out_path, PROJECT_DIR)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

CATEGORY_FILTER = {
    "it":       "INFORMATION-TECHNOLOGY",
    "designer": "DESIGNER",
}


def main():
    parser = argparse.ArgumentParser(description="Per-model stability analysis by demographic group.")
    parser.add_argument(
        "--category", choices=["it", "designer"], default=None,
        help="Filter to a single job category. Omit to run both.",
    )
    args = parser.parse_args()

    suffix    = f"_{args.category}" if args.category else ""
    cat_label = CATEGORY_LABELS.get(CATEGORY_FILTER.get(args.category, ""), args.category or "All")
    title_tag = f" ({cat_label})" if args.category else ""

    print(f"Loading {os.path.relpath(RAW_SCORES, PROJECT_DIR)} ...")
    rows = load_raw_scores(RAW_SCORES)
    print(f"  {len(rows):,} score records")

    if args.category:
        target = CATEGORY_FILTER[args.category]
        rows = [r for r in rows if r["category"] == target]
        print(f"  {len(rows):,} records after filtering to {target}")

    # --- score stability ---
    print("\nComputing per-resume score stability ...")
    stab_records = per_resume_stability(rows)
    print(f"  {len(stab_records):,} (model × resume × demographic) groups")

    print("\nAggregating race score stability by model ...")
    race_by_model = aggregate_by_model(stab_records, "race")
    save_csv(race_by_model,  os.path.join(ANALYSIS_DIR, f"race_stability_by_model{suffix}.csv"))
    save_json(race_by_model, os.path.join(ANALYSIS_DIR, f"race_stability_by_model{suffix}.json"))

    print("Aggregating gender score stability by model ...")
    gender_by_model = aggregate_by_model(stab_records, "gender")
    save_csv(gender_by_model,  os.path.join(ANALYSIS_DIR, f"gender_stability_by_model{suffix}.csv"))
    save_json(gender_by_model, os.path.join(ANALYSIS_DIR, f"gender_stability_by_model{suffix}.json"))

    # --- rank stability ---
    print("\nComputing rank stability (Spearman ρ across paraphrase versions) ...")

    print("  Race rank stability by model ...")
    race_rank_by_model = compute_rank_stability_by_group(rows, "race")
    save_csv(race_rank_by_model,  os.path.join(ANALYSIS_DIR, f"race_rank_stability_by_model{suffix}.csv"))
    save_json(race_rank_by_model, os.path.join(ANALYSIS_DIR, f"race_rank_stability_by_model{suffix}.json"))

    print("  Gender rank stability by model ...")
    gender_rank_by_model = compute_rank_stability_by_group(rows, "gender")
    save_csv(gender_rank_by_model,  os.path.join(ANALYSIS_DIR, f"gender_rank_stability_by_model{suffix}.csv"))
    save_json(gender_rank_by_model, os.path.join(ANALYSIS_DIR, f"gender_rank_stability_by_model{suffix}.json"))

    # --- plots ---
    print("\nGenerating plots ...")
    plot_stability_heatmap(
        race_by_model, race_rank_by_model, "race",
        f"Stability by Model and Race{title_tag}",
        os.path.join(PLOTS_DIR, f"stability_heatmap_race{suffix}.png"),
    )
    plot_stability_heatmap(
        gender_by_model, gender_rank_by_model, "gender",
        f"Stability by Model and Gender{title_tag}",
        os.path.join(PLOTS_DIR, f"stability_heatmap_gender{suffix}.png"),
    )
    plot_score_distribution_by_model(
        rows, "race",
        f"Score Distribution by Model and Race{title_tag}",
        os.path.join(PLOTS_DIR, f"score_dist_by_race{suffix}.png"),
    )
    plot_score_distribution_by_model(
        rows, "gender",
        f"Score Distribution by Model and Gender{title_tag}",
        os.path.join(PLOTS_DIR, f"score_dist_by_gender{suffix}.png"),
    )

    # --- console summary ---
    print("\n=== Race Score Stability by Model ===")
    for r in race_by_model:
        print(f"  {r['model_short']:20s} | {CATEGORY_LABELS.get(r['category'], r['category']):10s} | "
              f"{r['race']:30s} | score_stab={r['avg_stability']:.4f}  (n={r['count']})")

    print("\n=== Race Rank Stability by Model ===")
    for r in race_rank_by_model:
        print(f"  {r['model_short']:20s} | {CATEGORY_LABELS.get(r['category'], r['category']):10s} | "
              f"{r['race']:30s} | rank_stab={r['avg_rank_stability']:.4f}  "
              f"(n_resumes={r['n_resumes']}, n_pairs={r['n_version_pairs']})")

    print("\n=== Gender Score Stability by Model ===")
    for r in gender_by_model:
        print(f"  {r['model_short']:20s} | {CATEGORY_LABELS.get(r['category'], r['category']):10s} | "
              f"{r['gender']:6s} | score_stab={r['avg_stability']:.4f}  (n={r['count']})")

    print("\n=== Gender Rank Stability by Model ===")
    for r in gender_rank_by_model:
        print(f"  {r['model_short']:20s} | {CATEGORY_LABELS.get(r['category'], r['category']):10s} | "
              f"{r['gender']:6s} | rank_stab={r['avg_rank_stability']:.4f}  "
              f"(n_resumes={r['n_resumes']}, n_pairs={r['n_version_pairs']})")


if __name__ == "__main__":
    main()
