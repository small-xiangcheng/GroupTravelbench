"""
Evaluation-report generator.

Merge ``multi_eval`` and ``llm_judge`` results into a readable Markdown
report. Core sections:
1. Overview table (5 dimensions × Overall/Easy/Medium/Hard)
2. Detailed rule-based results (including LLM-Judge Average)
3. LLM-Judge per-dimension scores
4. Execution statistics
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from collections import defaultdict


def generate_report(
    multi_eval_output: Dict[str, Any],
    llm_judge_output: Optional[Dict[str, Any]] = None,
    agent_name: str = "Unknown",
    output_path: Optional[str] = None,
) -> str:
    """
    Generate a complete Markdown evaluation report.

    Args:
        multi_eval_output: Return value of ``multi_eval.evaluate_batch()``.
        llm_judge_output: Optional LLM-Judge JSON output.
        agent_name: Agent name used in the report title.
        output_path: Optional path to write the report.

    Returns:
        Report string in Markdown.
    """
    lines = []

    # Header
    lines.append(f"# Evaluation Report: {agent_name}")
    lines.append(f"> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    tool_stats = multi_eval_output.get("tool_stats", {})
    total_tasks = multi_eval_output.get("total_tasks", 0)
    total_raw = multi_eval_output.get("total_raw_samples", 0)
    sample_count = tool_stats.get("sample_count", 1)

    overall_summary = multi_eval_output.get("overall_summary", {})
    difficulty_summaries = multi_eval_output.get("difficulty_summaries", {})

    # Compute LLM-Judge scores by difficulty (including sub-dimensions)
    llm_judge_by_difficulty = _compute_llm_judge_by_difficulty(
        llm_judge_output, multi_eval_output
    )

    # Precompute per-difficulty column info
    difficulty_order = ["easy", "medium", "hard"]
    col_keys = ["overall"] + [d for d in difficulty_order if d in difficulty_summaries]
    summaries = {"overall": overall_summary}
    counts = {"overall": total_tasks}
    for diff in difficulty_order:
        if diff in difficulty_summaries:
            summaries[diff] = difficulty_summaries[diff].get("summary", {})
            counts[diff] = difficulty_summaries[diff].get("count", 0)

    # ======================================================================
    # Table 1: overview (2 rows) — Group Utility + mean of the other 4 dimensions
    # ======================================================================
    lines.append("## 📊 Score Overview")
    lines.append("")
    lines.append(_build_overview_table(col_keys, counts, summaries, llm_judge_by_difficulty))
    lines.append("")

    # ======================================================================
    # Table 2: per-dimension breakdown (5 rows)
    # ======================================================================
    lines.append("## 📈 Scores by Dimension")
    lines.append("")
    lines.append(_build_dimension_table(col_keys, counts, summaries, llm_judge_by_difficulty))
    lines.append("")

    # ======================================================================
    # Table 3: LLM-Judge sub-dimensions
    # ======================================================================
    if llm_judge_output:
        statistics = llm_judge_output.get("statistics", {})
        lines.append("## 🤖 LLM-Judge Sub-dimensions")
        lines.append("")
        lines.append(f"- **Model**: {llm_judge_output.get('config', {}).get('model', 'N/A')}")
        lines.append(f"- **Evaluated**: {statistics.get('total_evaluated', 0)} samples")
        if statistics.get("meta_judge"):
            meta = statistics["meta_judge"]
            lines.append(f"- **Meta-Judge**: {meta.get('mean', 0):.2f}/5 (n={meta.get('count', 0)})")
        lines.append("")
        lines.append(_build_llm_judge_table(col_keys, llm_judge_by_difficulty, statistics))
        lines.append("")

    # ======================================================================
    # Section 4: execution statistics
    # ======================================================================
    lines.append("## 🔧 Execution Statistics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|------:|")
    lines.append(f"| Total Tasks | {total_tasks} |")
    lines.append(f"| Total Samples (with trials) | {total_raw} |")
    lines.append(f"| Plan Generated Rate | {tool_stats.get('plan_generated_rate', 0):.2%} |")
    lines.append(f"| Avg Duration (s) | {tool_stats.get('avg_duration', 0):.2f} |")
    lines.append(f"| Avg Tool Calls / Sample | {tool_stats.get('avg_tool_calls', 0):.2f} |")
    cache_total = tool_stats.get("total_cache_hits", 0) + tool_stats.get("total_cache_misses", 0)
    hit_rate = tool_stats.get("total_cache_hits", 0) / cache_total if cache_total > 0 else 0
    lines.append(f"| Cache Hit Rate | {hit_rate:.2%} |")
    lines.append(f"| Tool Error Rate | {tool_stats.get('tool_error_rate', 0):.2%} |")
    lines.append("")

    # Completion Reasons
    completion_reasons = tool_stats.get("completion_reasons", {})
    if completion_reasons:
        lines.append("### Completion Reasons")
        lines.append("")
        lines.append("| Reason | Count | Ratio |")
        lines.append("|--------|------:|------:|")
        for reason, count in sorted(completion_reasons.items(), key=lambda x: -x[1]):
            ratio = count / sample_count if sample_count > 0 else 0
            lines.append(f"| {reason} | {count} | {ratio:.2%} |")
        lines.append("")

    report_text = "\n".join(lines)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_text)

    return report_text


# =============================================================================
# Table builders
# =============================================================================

def _col_header(col_keys: List[str], counts: Dict[str, int]) -> List[str]:
    """Build a table column header."""
    return [f"Overall ({counts.get('overall', 0)})" if k == "overall"
            else f"{k.capitalize()} ({counts.get(k, 0)})"
            for k in col_keys]


def _build_overview_table(
    col_keys: List[str],
    counts: Dict[str, int],
    summaries: Dict[str, Any],
    llm_judge_by_difficulty: Dict[str, Dict[str, float]],
) -> str:
    """
    Overview table (2 rows):
    - Group Utility (per-user avg)
    - Other Dimensions Average (mean of PC + GF + PV + LLM-Judge Avg)
    """
    header_cols = _col_header(col_keys, counts)
    header = "| Metric | " + " | ".join(header_cols) + " |"
    separator = "|--------|" + "|".join([":------:" for _ in header_cols]) + "|"

    # Row 1: Group Utility
    gu_values = []
    for col_key in col_keys:
        s = summaries.get(col_key, {}).get("group_utility", {})
        val = s.get("average_score", 0)
        gu_values.append(f"**{val:.2f}**")

    # Row 2: Other Dimensions Average (PC + GF + PV + LLM-Judge Avg) / 4
    other_avg_values = []
    for col_key in col_keys:
        pc = summaries.get(col_key, {}).get("preference_completeness", {}).get("average_ratio", 0) * 100
        gf = summaries.get(col_key, {}).get("group_fairness", {}).get("average_fairness_score", 0)
        pv = summaries.get(col_key, {}).get("plan_validity", {}).get("valid_ratio", 0) * 100
        llm = llm_judge_by_difficulty.get(col_key, {}).get("average_score", 0)
        avg = (pc + gf + pv + llm) / 4
        other_avg_values.append(f"**{avg:.2f}**")

    rows = [
        "| **Group Utility (per-user avg)** | " + " | ".join(gu_values) + " |",
        "| **Other Dimensions Avg** | " + " | ".join(other_avg_values) + " |",
    ]

    return "\n".join([header, separator] + rows)


def _build_dimension_table(
    col_keys: List[str],
    counts: Dict[str, int],
    summaries: Dict[str, Any],
    llm_judge_by_difficulty: Dict[str, Dict[str, float]],
) -> str:
    """
    Per-dimension table (5 rows):
    - Preference Completeness (0-100%)
    - Group Utility per-user avg (higher=better)
    - Group Fairness (0-100%)
    - Plan Validity (0-100%)
    - LLM-Judge Average (0-100)
    """
    header_cols = _col_header(col_keys, counts)
    header = "| Dimension | " + " | ".join(header_cols) + " |"
    separator = "|-----------|" + "|".join([":------:" for _ in header_cols]) + "|"

    rows = []

    # Preference Completeness
    vals = []
    for col_key in col_keys:
        s = summaries.get(col_key, {}).get("preference_completeness", {})
        vals.append(f"{s.get('average_ratio', 0) * 100:.2f}")
    rows.append("| Preference Completeness | " + " | ".join(vals) + " |")

    # Group Utility
    vals = []
    for col_key in col_keys:
        s = summaries.get(col_key, {}).get("group_utility", {})
        vals.append(f"{s.get('average_score', 0):.2f}")
    rows.append("| Group Utility (per-user avg) | " + " | ".join(vals) + " |")

    # Group Fairness
    vals = []
    for col_key in col_keys:
        s = summaries.get(col_key, {}).get("group_fairness", {})
        vals.append(f"{s.get('average_fairness_score', 0):.2f}")
    rows.append("| Group Fairness | " + " | ".join(vals) + " |")

    # Plan Validity
    vals = []
    for col_key in col_keys:
        s = summaries.get(col_key, {}).get("plan_validity", {})
        vals.append(f"{s.get('valid_ratio', 0) * 100:.2f}")
    rows.append("| Plan Validity | " + " | ".join(vals) + " |")

    # LLM-Judge Average
    vals = []
    for col_key in col_keys:
        llm = llm_judge_by_difficulty.get(col_key, {})
        vals.append(f"{llm.get('average_score', 0):.2f}" if llm else "-")
    rows.append("| LLM-Judge Average | " + " | ".join(vals) + " |")

    return "\n".join([header, separator] + rows)


def _build_llm_judge_table(
    col_keys: List[str],
    llm_judge_by_difficulty: Dict[str, Dict[str, float]],
    statistics: Dict[str, Any],
) -> str:
    """
    LLM-Judge's 5 sub-dimensions + Average × Overall/Easy/Medium/Hard.

    Always uses ``llm_judge_by_difficulty`` (already task-averaged and
    meta-judge weighted).
    """
    dimensions = statistics.get("dimensions", [])

    header_cols = ["Overall"] + [k.capitalize() for k in col_keys if k != "overall"]
    header = "| Dimension | " + " | ".join(header_cols) + " |"
    separator = "|-----------|" + "|".join([":------:" for _ in header_cols]) + "|"

    rows = []
    for dim in dimensions:
        dim_display = dim.replace("_", " ").title()
        values = []
        for col_key in col_keys:
            diff_scores = llm_judge_by_difficulty.get(col_key, {})
            val = diff_scores.get(dim, 0)
            values.append(f"{val:.2f}")
        rows.append(f"| {dim_display} | " + " | ".join(values) + " |")

    # Average row
    avg_values = []
    for col_key in col_keys:
        diff_avg = llm_judge_by_difficulty.get(col_key, {}).get("average_score", 0)
        avg_values.append(f"**{diff_avg:.2f}**")
    rows.append("| **Average** | " + " | ".join(avg_values) + " |")

    return "\n".join([header, separator] + rows)


# =============================================================================
# LLM-Judge score computation
# =============================================================================

def _compute_llm_judge_by_difficulty(
    llm_judge_output: Optional[Dict[str, Any]],
    multi_eval_output: Dict[str, Any],
) -> Dict[str, Dict[str, float]]:
    """
    Compute LLM-Judge scores (including sub-dimensions) by difficulty.

    Average trials of the same ``task_id`` first, then average within each
    difficulty bucket. Tasks that were never scored count as 0.
    """
    result: Dict[str, Dict[str, float]] = {}

    if not llm_judge_output:
        return result

    details = llm_judge_output.get("details", [])
    if not details:
        return result

    # Collect the dimension list
    statistics = llm_judge_output.get("statistics", {})
    dimensions = statistics.get("dimensions", [])

    # Build task_id -> difficulty_type from multi_eval details
    difficulty_map = {}
    all_task_ids = set()
    for item in multi_eval_output.get("details", []):
        task_id = item.get("task_id", "")
        difficulty = item.get("difficulty_type", "unknown")
        if task_id:
            difficulty_map[task_id] = difficulty
            all_task_ids.add(task_id)

    # Group LLM-Judge scores by task_id (average + sub-dimensions)
    task_avg_scores: Dict[str, List[float]] = defaultdict(list)
    task_dim_scores: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for item in details:
        evaluation = item.get("evaluation", {})
        final_scores = evaluation.get("final_scores", {})
        avg_score = final_scores.get("average_score")
        if avg_score is None:
            continue
        task_id = item.get("task_id", "")
        if not task_id:
            continue
        task_avg_scores[task_id].append(avg_score)
        for dim in dimensions:
            dim_score = final_scores.get(f"{dim}_score")
            if dim_score is not None:
                task_dim_scores[task_id][dim].append(dim_score)

    # Average per task_id
    task_averaged: Dict[str, Dict[str, float]] = {}
    for task_id in all_task_ids:
        entry: Dict[str, float] = {}
        scores = task_avg_scores.get(task_id, [])
        entry["average_score"] = round(sum(scores) / len(scores), 2) if scores else 0.0
        for dim in dimensions:
            dim_scores_list = task_dim_scores.get(task_id, {}).get(dim, [])
            entry[dim] = round(sum(dim_scores_list) / len(dim_scores_list), 2) if dim_scores_list else 0.0
        task_averaged[task_id] = entry

    # Group by difficulty
    groups: Dict[str, List[Dict[str, float]]] = {"overall": []}
    for task_id, scores_dict in task_averaged.items():
        groups["overall"].append(scores_dict)
        difficulty = difficulty_map.get(task_id, "unknown")
        if difficulty not in groups:
            groups[difficulty] = []
        groups[difficulty].append(scores_dict)

    # Average each group
    for group_name, score_dicts in groups.items():
        if not score_dicts:
            continue
        group_avg: Dict[str, float] = {}
        group_avg["average_score"] = round(sum(d["average_score"] for d in score_dicts) / len(score_dicts), 2)
        for dim in dimensions:
            dim_vals = [d.get(dim, 0) for d in score_dicts]
            group_avg[dim] = round(sum(dim_vals) / len(dim_vals), 2) if dim_vals else 0.0
        result[group_name] = group_avg

    return result
