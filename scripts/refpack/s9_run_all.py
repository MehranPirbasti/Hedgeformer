"""
Section 9 of the work plan: what to run, in what order, and what each step
buys.

This module does not hide the cost of the programme. It prints the run count
per phase so a decision about compute budget can be made with the numbers
visible, and it emits the manuscript tables from whatever results exist so
that a partial programme still produces a paper.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .config import (DATA_FRACTIONS, REPORT_DIR, RUN_DIR, SEEDS, VARIANTS,
                     VARIANT_LABEL)

# s2_protocol, s3_baselines and s4_ablation pull in torch, which is not needed
# to print the plan or emit the checklist, so they are imported lazily.


@dataclass
class Task:
    key: str
    section: str
    title: str
    phase: int
    n_runs: int                 # training runs required, 0 if analysis only
    blocks_submission: bool
    rationale: str
    entry_point: str


TASKS: List[Task] = [
    # ---- phase 1, nothing ships without these -----------------------------
    Task("data_audit", "1", "Lock the data facts", 1, 0, True,
         "Provenance of the elevation layer, acquisition dates, "
         "co-registration and class support. The physical argument in the "
         "introduction is unsupported until the dates exist in the paper.",
         "refpack.s1_data_audit"),
    Task("controlled_runs", "2.1/2.3", "Rerun all variants under one protocol",
         1, len(VARIANTS) * len(SEEDS), True,
         "Equal epochs, one checkpoint criterion, three seeds. The current "
         "margins sit in the fourth decimal and cannot be defended from a "
         "single seed.",
         "refpack.s2_protocol.run_matrix"),
    Task("data_fraction", "2.2", "Data-efficiency ablation", 1,
         len(VARIANTS) * len(SEEDS) * (len(DATA_FRACTIONS) - 1), False,
         "Turns the unequal data volume from a flaw into the strongest claim "
         "available: the adaptive operator leads by more when data is scarce.",
         "refpack.s2_protocol.build_stratified_subsets"),
    Task("baselines", "3", "External baselines", 1, 4 * len(SEEDS), True,
         "Single-modality, early fusion, late fusion and a classical "
         "non-learning method. Without these there is no claim of state of "
         "the art, only an internal comparison.",
         "refpack.s3_baselines"),
    Task("loss_ablation", "4", "Loss ablation", 1, 4, True,
         "Four loss components are claimed as contributions and none is "
         "ablated.",
         "refpack.s4_ablation.ABLATION_GRID"),
    Task("complexity", "2.4", "Parameters and inference cost", 1, 0, True,
         "Three FILL entries in the complexity table, plus the seconds per "
         "square kilometre that the national-monitoring motivation implies.",
         "refpack.s2_protocol.parameter_report"),
    Task("ordinal", "6.1", "Ordinal metrics and confusion matrix", 1, 0, True,
         "Height error in metres is the number that makes the results legible "
         "next to the canopy-height literature and defuses the objection to a "
         "macro F1 of 0.43.",
         "refpack.s6_metrics.ordinal_metrics"),
    Task("match_rate", "9", "Report the instance match rate", 1, 0, True,
         "Matched-instance geometry improves trivially for a model that "
         "detects fewer and easier instances. The match rate is already "
         "computed as connectivity correctness; it only has to be reported "
         "next to every distance metric.",
         "validation.FullMetrics"),

    # ---- phase 2, good to reference ---------------------------------------
    Task("modality_dropout", "5.2", "Modality ablation at inference", 2, 0, False,
         "Proves the mechanism directly and costs no training. The predicted "
         "cross-over between class 1a and class 6 is the result to look for.",
         "refpack.s5_mechanism.modality_ablation"),
    Task("weight_probe", "5.1", "Fast-weight probe", 2, 0, False,
         "The signature figure. Fast-weight movement against canopy signal to "
         "noise, contrasted with the frozen baseline.",
         "refpack.s5_mechanism.run_fast_weight_probe"),
    Task("phenology", "5.3", "Synthetic phenology stress", 2, 0, False,
         "Measures the leaf-off to leaf-on mismatch that motivates the whole "
         "architecture instead of asserting it.",
         "refpack.s5_mechanism.phenology_curve"),
    Task("ceiling", "6.3", "Achievable ceiling", 2, 0, False,
         "A general tool other groups will borrow, and the permanent answer "
         "to reviewers who object to absolute scores.",
         "refpack.s6_metrics.achievable_ceiling"),
    Task("coarea_length", "7.1", "Threshold-free length estimator", 2, 0, False,
         "Length straight from the proximity field, with no threshold and no "
         "skeletonisation.",
         "refpack.s7_length.length_from_field"),

    # ---- phase 3, what makes it last --------------------------------------
    Task("capture_recapture", "7.3", "True network length", 3, 0, False,
         "The first defensible estimate of how much hedgerow actually exists "
         "in this landscape, and what share the reference product sees.",
         "refpack.s7_length.network_length_report"),
    Task("connectivity", "6.5", "Landscape connectivity retention", 3, 0, False,
         "Connects segmentation quality to ecological function.",
         "refpack.s6_metrics.connectivity_retention"),
    Task("carbon", "6.6", "Carbon bias of a binary map", 3, 0, False,
         "The sentence that earns citations outside computer vision. Needs "
         "sourced biomass coefficients first.",
         "refpack.s6_metrics.binary_map_carbon_bias"),
    Task("benchmark", "8", "Feature-level benchmark release", 3, 0, False,
         "Publishable under the Airbus licence and, as far as we can tell, "
         "without precedent in this field.",
         "refpack.s8_benchmark.export_features"),
]


HIGHEST_RETURN = ["modality_dropout", "ordinal", "ceiling"]


def plan(phase: Optional[int] = None) -> Dict:
    tasks = [t for t in TASKS if phase is None or t.phase == phase]
    return {
        "n_tasks": len(tasks),
        "training_runs": sum(t.n_runs for t in tasks),
        "blocking": [t.key for t in tasks if t.blocks_submission],
        "tasks": [
            {"key": t.key, "section": t.section, "title": t.title,
             "phase": t.phase, "runs": t.n_runs,
             "blocks_submission": t.blocks_submission,
             "why": t.rationale, "entry_point": t.entry_point}
            for t in tasks
        ],
    }


def budget_report() -> Dict:
    full = sum(t.n_runs for t in TASKS)
    minimal = (len(VARIANTS) * len(SEEDS)                      # controlled runs
               + len(VARIANTS) * len(SEEDS)                    # one extra fraction
               + 4 * len(SEEDS)                                # baselines
               + 4)                                            # priority ablations
    return {
        "full_programme_runs": full,
        "minimal_defensible_runs": minimal,
        "note": ("The minimal programme drops the middle data fraction, keeps "
                 "two fractions, four baselines and four loss ablations. It "
                 "still supports every claim the paper makes."),
        "highest_return_if_time_is_short": HIGHEST_RETURN,
    }


# --------------------------------------------------------------------------
# manuscript table emission
# --------------------------------------------------------------------------
def emit_seed_table(summaries: Dict[str, Dict], metrics: List[str],
                    path: str, caption: str, label: str) -> str:
    """LaTeX table with mean and standard deviation over seeds."""
    lines = [
        r"\begin{table}[!t]", r"\centering",
        f"\\caption{{{caption}}}", f"\\label{{{label}}}",
        r"\setlength{\tabcolsep}{4pt}", r"\renewcommand{\arraystretch}{1.12}",
        r"\begin{tabular}{@{}l ccc@{}}", r"\toprule",
        "Metric & " + " & ".join(VARIANT_LABEL[v].replace(" ", "~")
                                 for v in VARIANTS) + r" \\",
        r"\midrule",
    ]
    for m in metrics:
        row = summaries.get(m, {})
        cells = []
        best = max((row[v].mean for v in VARIANTS if v in row), default=None)
        for v in VARIANTS:
            if v not in row:
                cells.append("--")
                continue
            s = row[v]
            cell = s.cell()
            if best is not None and abs(s.mean - best) < 1e-12:
                cell = r"\textbf{" + cell + "}"
            cells.append(cell)
        lines.append(f"{m.replace('_', ' ')} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def emit_checklist(path: str) -> str:
    """A markdown checklist to paste into the project tracker."""
    lines = ["# HedgeFormer, reference-grade checklist", ""]
    for ph in (1, 2, 3):
        head = {1: "Phase 1, blocking", 2: "Phase 2, reference grade",
                3: "Phase 3, lasting"}[ph]
        lines += [f"## {head}", ""]
        for t in TASKS:
            if t.phase != ph:
                continue
            runs = f" ({t.n_runs} training runs)" if t.n_runs else ""
            flag = " **blocks submission**" if t.blocks_submission else ""
            lines += [f"- [ ] **{t.title}**, section {t.section}{runs}{flag}",
                      f"      {t.rationale}",
                      f"      entry point: `{t.entry_point}`", ""]
    b = budget_report()
    lines += ["## Budget", "",
              f"- full programme: {b['full_programme_runs']} training runs",
              f"- minimal defensible programme: {b['minimal_defensible_runs']} runs",
              f"- highest return if time is short: "
              f"{', '.join(b['highest_return_if_time_is_short'])}", ""]
    with open(path, "w") as fh:
        fh.write("\n".join(lines))
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, default=None)
    ap.add_argument("--out", default=REPORT_DIR)
    ap.add_argument("--emit-scripts", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    p = plan(args.phase)
    b = budget_report()

    with open(os.path.join(args.out, "s9_plan.json"), "w") as fh:
        json.dump({"plan": p, "budget": b}, fh, indent=2)
    emit_checklist(os.path.join(args.out, "CHECKLIST.md"))

    if args.emit_scripts:
        from . import s2_protocol
        rows = s2_protocol.run_matrix()
        s2_protocol.emit_run_script(rows, os.path.join(args.out, "run_matrix.sh"))
        s2_protocol.emit_run_script(s2_protocol.minimal_run_matrix(),
                                    os.path.join(args.out, "run_matrix_minimal.sh"))

    print(f"tasks: {p['n_tasks']}  training runs in this selection: {p['training_runs']}")
    print(f"full programme: {b['full_programme_runs']} runs, "
          f"minimal: {b['minimal_defensible_runs']} runs")
    print("blocking:", ", ".join(p["blocking"]))
    print("written to", args.out)


if __name__ == "__main__":
    main()


# ==========================================================================
# fast-track scheduler for a hard deadline
# ==========================================================================
def _run_hours(kind: str, patches: int, epochs: int, early_stop: bool = True) -> float:
    from .config import (THROUGHPUT, THROUGHPUT_SINGLE_ENCODER, VAL_MINUTES,
                         EARLY_STOP_SAVING)
    if kind in THROUGHPUT:
        tp, val = THROUGHPUT[kind], VAL_MINUTES[kind]
    else:
        tp, val = THROUGHPUT_SINGLE_ENCODER, VAL_MINUTES["single"]
    per_epoch_h = patches / tp / 3600.0 + val / 60.0
    h = per_epoch_h * epochs
    return h * (1.0 - EARLY_STOP_SAVING) if early_stop else h


def fast_track_schedule(start: str = "2026-09-21", deadline: str = "2026-10-14",
                        writing_days: int = 4, gpu_hours_per_day: float = 22.0) -> Dict:
    """
    An ordered GPU queue in which every stopping point leaves a submittable
    paper. Items are ordered by value per GPU hour, not by section number.

    Nothing is retrained if an existing checkpoint already satisfies the
    controlled protocol. The concatenation run qualifies: it trained on the
    full set and its best-F1 checkpoint fell at epoch 14, inside the 30-epoch
    cap, so it is exactly what a fresh capped run would have selected.
    """
    from datetime import date, timedelta
    from .config import FAST_EPOCH_CAP as E, FULL_TRAIN_PATCHES as FULL, \
        SMALL_TRAIN_PATCHES as SMALL

    q = [
        ("inference_only", "Mechanism, metrics, length, ceiling, classical "
         "baseline, significance on existing checkpoints", 10.0,
         "Every novel analysis in the paper. Sections 5, 6, 7, the classical "
         "baseline of 3e and patch-level bootstrap all run here."),
        ("early_fusion", "Early-fusion baseline, full data",
         _run_hours("single", FULL, E),
         "The external baseline a reviewer checks first."),
        ("ttt_full", "TTT rerun on full data",
         _run_hours("TTT", FULL, E),
         "Closes the unequal-data objection for the proposed model."),
        ("sra_full", "SRA rerun on full data, selected on F1",
         _run_hours("SRA", FULL, E),
         "Closes the mixed-data and wrong-selection-criterion objections."),
        ("ms_only", "Spectral-only baseline, full data",
         _run_hours("single", FULL, E),
         "Half of the per-class asymmetry that justifies fusion."),
        ("dem_only", "Elevation-only baseline, full data",
         _run_hours("single", FULL, E),
         "The other half. Together they turn the introduction into a result."),
        ("concat_small", "Concatenation on the reduced set",
         _run_hours("Simple", SMALL, E),
         "Completes the data-efficiency comparison. The existing TTT and SRA "
         "runs already supply the reduced-data point for those two."),
        ("loss_ablation", "Three loss ablations on TTT, 25 percent data",
         3 * _run_hours("TTT", int(FULL * 0.25), E),
         "no_pu, no_merge, textbook_cldice. The silence probe covers the "
         "clDice corrections for free in the meantime."),
        ("seeds", "Two further seeds for all three variants, full data",
         2 * (_run_hours("Simple", FULL, E) + _run_hours("SRA", FULL, E)
              + _run_hours("TTT", FULL, E)),
         "Seed variance. Until this finishes, patch-level bootstrap intervals "
         "carry the uncertainty and the paper says so."),
    ]

    d0 = date.fromisoformat(start)
    dl = date.fromisoformat(deadline)
    gpu_days = (dl - d0).days - writing_days
    budget_h = gpu_days * gpu_hours_per_day

    rows, cum = [], 0.0
    for key, title, hours, why in q:
        cum += hours
        finish = d0 + timedelta(days=cum / gpu_hours_per_day)
        rows.append({"key": key, "title": title, "hours": round(hours, 1),
                     "cumulative_hours": round(cum, 1),
                     "finishes": finish.isoformat(),
                     "fits": cum <= budget_h, "why": why})

    return {"start": start, "deadline": deadline,
            "gpu_days_available": gpu_days,
            "gpu_hours_available": round(budget_h, 1),
            "queue": rows,
            "total_hours": round(cum, 1),
            "total_days": round(cum / gpu_hours_per_day, 1),
            "margin_hours": round(budget_h - cum, 1)}
