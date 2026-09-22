"""
Unified entry point for multi-user travel-planning evaluation.

Combines preference completeness, group utility, group fairness, and
plan validity, with APIs for a single result or a batch file.

Supports:
- Multi-trial averaging (group by ``task_id``, mean of trial scores)
- Breakdown by ``difficulty_type`` (easy/medium/hard)
- Tool-call statistics (cache_hits, cache_misses, tool_errors, ...)

Usage:
    python -m group_travelbench.evaluation.multi_eval \
        --input /path/to/result.jsonl \
        --output /path/to/eval_output.json \
        --test-file /path/to/test.jsonl
"""

import json
import re
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import defaultdict

from .preference_completeness import evaluate_preference_completeness
from .group_utility import evaluate_group_utility, extract_plan_from_conversation
from .group_fairness import evaluate_group_fairness
from .plan_validity import evaluate_plan_validity


def evaluate_single_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run the full evaluation on a single result.

    Args:
        result: One complete result dict (one JSONL line).

    Returns:
        Dict with the four evaluation dimensions.
    """
    task_id = result.get("task_id", "unknown")
    trial_id = result.get("trial_id", 0)
    plan_generated = result.get("plan_generated", False)

    eval_result = {
        "task_id": task_id,
        "trial_id": trial_id,
        "plan_generated": plan_generated,
        "preference_completeness": None,
        "group_utility": None,
        "group_fairness": None,
        "plan_validity": None,
    }

    # Extract basic fields
    agent_pref_table = result.get("agent_preference_table", {})
    effective_preferences = result.get("effective_preferences", {})
    user_names = result.get("user_names", [])
    conversation_history = result.get("conversation_history", [])

    # Intercept inference-crash samples: when a run fails (success=False, e.g.
    # context_overflow / rate_limit / connection_error / runtime_error）
    # the output has no effective_preferences (the reference), so preference-completeness
    # would have total_possible = 0 and recall would become 1.0 — that would disguise
    # a crash as 'perfect preference collection' and inflate the metric. All dimensions
    # of such a sample are recorded as error (averaged as 0), matching group_utility /
    # fairness / plan_validity.
    inference_failed = (result.get("success") is False) or (not effective_preferences)
    if inference_failed:
        error_reason = "inference failed (no effective_preferences)"
        eval_result["preference_completeness"] = {"error": error_reason}
        eval_result["group_utility"] = {"error": error_reason}
        eval_result["group_fairness"] = {"error": error_reason}
        eval_result["plan_validity"] = {"error": error_reason}
        return eval_result

    # 1. Preference completeness (scorable even without a plan)
    eval_result["preference_completeness"] = evaluate_preference_completeness(
        agent_pref_table, effective_preferences
    )

    # 2-4: require a plan
    if not plan_generated:
        eval_result["group_utility"] = {"error": "plan not generated"}
        eval_result["group_fairness"] = {"error": "plan not generated"}
        eval_result["plan_validity"] = {"error": "plan not generated"}
        return eval_result

    plan = extract_plan_from_conversation(conversation_history)
    if plan is None:
        eval_result["group_utility"] = {"error": "failed to extract plan from conversation"}
        eval_result["group_fairness"] = {"error": "failed to extract plan from conversation"}
        eval_result["plan_validity"] = {"error": "failed to extract plan from conversation"}
        return eval_result

    # 2. Group utility
    eval_result["group_utility"] = evaluate_group_utility(
        plan, effective_preferences, user_names
    )

    # 3. Group fairness
    eval_result["group_fairness"] = evaluate_group_fairness(
        plan, effective_preferences, user_names
    )

    # 4. Plan validity
    eval_result["plan_validity"] = evaluate_plan_validity(plan, user_names)

    return eval_result


def _load_difficulty_map(test_file: Optional[str] = None) -> Dict[str, str]:
    """Load a ``task_id -> difficulty_type`` map from ``test.jsonl``."""
    if not test_file:
        return {}
    test_path = Path(test_file)
    if not test_path.exists():
        return {}
    difficulty_map = {}
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            task_id = data.get("task_id", "")
            difficulty = data.get("difficulty_type", "unknown")
            if task_id:
                difficulty_map[task_id] = difficulty
    return difficulty_map


def _average_trial_results(eval_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Group by ``task_id`` and average scores across trials.

    Returns:
        One record per ``task_id``, containing the averaged evaluation.
    """
    task_groups: Dict[str, List[Dict]] = defaultdict(list)
    for result in eval_results:
        task_id = result.get("task_id", "unknown")
        task_groups[task_id].append(result)

    averaged_results = []
    for task_id, trials in task_groups.items():
        if len(trials) == 1:
            averaged_results.append(trials[0])
            continue

        # Average across trials
        averaged = {
            "task_id": task_id,
            "plan_generated": any(t.get("plan_generated", False) for t in trials),
            "num_trials": len(trials),
        }

        # Average preference_completeness
        pc_ratios = [
            t["preference_completeness"]["completeness_ratio"]
            for t in trials
            if t.get("preference_completeness") and "completeness_ratio" in t["preference_completeness"]
        ]
        if pc_ratios:
            averaged["preference_completeness"] = {
                "completeness_ratio": sum(pc_ratios) / len(pc_ratios),
                "averaged_from": len(pc_ratios),
            }
        else:
            averaged["preference_completeness"] = None

        # Average group_utility
        gu_scores = [
            t["group_utility"]["total_score"]
            for t in trials
            if t.get("group_utility") and "total_score" in t["group_utility"]
        ]
        gu_per_user_scores = [
            t["group_utility"]["per_user_average_score"]
            for t in trials
            if t.get("group_utility") and "per_user_average_score" in t["group_utility"]
        ]
        if gu_scores:
            averaged["group_utility"] = {
                "total_score": sum(gu_scores) / len(gu_scores),
                "per_user_average_score": (
                    sum(gu_per_user_scores) / len(gu_per_user_scores)
                    if gu_per_user_scores
                    else sum(gu_scores) / len(gu_scores)
                ),
                "averaged_from": len(gu_scores),
            }
        else:
            averaged["group_utility"] = {"error": "no valid trials"}

        # Average group_fairness
        gf_scores = [
            t["group_fairness"]["fairness_score"]
            for t in trials
            if t.get("group_fairness") and "fairness_score" in t["group_fairness"]
        ]
        if gf_scores:
            averaged["group_fairness"] = {
                "fairness_score": sum(gf_scores) / len(gf_scores),
                "averaged_from": len(gf_scores),
            }
        else:
            averaged["group_fairness"] = {"error": "no valid trials"}

        # Average plan_validity
        pv_valids = [
            t["plan_validity"]["valid"]
            for t in trials
            if t.get("plan_validity") and "valid" in t["plan_validity"]
        ]
        pv_issues = [
            t["plan_validity"]["total_issues"]
            for t in trials
            if t.get("plan_validity") and "total_issues" in t["plan_validity"]
        ]
        if pv_valids:
            averaged["plan_validity"] = {
                "valid_ratio_across_trials": sum(pv_valids) / len(pv_valids),
                "total_issues": sum(pv_issues) / len(pv_issues) if pv_issues else 0,
                "averaged_from": len(pv_valids),
            }
        else:
            averaged["plan_validity"] = {"error": "no valid trials"}

        averaged_results.append(averaged)

    return averaged_results


def _collect_tool_stats(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collect tool-call statistics from inference outputs."""
    total_cache_hits = 0
    total_cache_misses = 0
    total_tool_errors = 0
    total_tool_calls = 0
    total_duration = 0.0
    plan_generated_count = 0
    completion_reasons: Dict[str, int] = defaultdict(int)
    sample_count = 0

    for result in results:
        sample_count += 1
        stats = result.get("agent_tool_stats", {})
        total_cache_hits += stats.get("cache_hits", 0)
        total_cache_misses += stats.get("cache_misses", 0)
        total_tool_errors += stats.get("tool_errors", 0)
        total_tool_calls += stats.get("tool_calls_executed", 0)
        total_duration += result.get("duration", 0.0)
        if result.get("plan_generated", False):
            plan_generated_count += 1
        reason = result.get("completion_reason", "unknown")
        completion_reasons[reason] += 1

    cache_total = total_cache_hits + total_cache_misses
    return {
        "sample_count": sample_count,
        "plan_generated_count": plan_generated_count,
        "plan_generated_rate": round(plan_generated_count / sample_count, 2) if sample_count > 0 else 0,
        "total_duration": round(total_duration, 2),
        "avg_duration": round(total_duration / sample_count, 2) if sample_count > 0 else 0,
        "total_tool_calls": total_tool_calls,
        "avg_tool_calls": round(total_tool_calls / sample_count, 2) if sample_count > 0 else 0,
        "total_cache_hits": total_cache_hits,
        "total_cache_misses": total_cache_misses,
        "cache_hit_rate": round(total_cache_hits / cache_total, 2) if cache_total > 0 else 0,
        "total_tool_errors": total_tool_errors,
        "tool_error_rate": round(total_tool_errors / total_tool_calls, 2) if total_tool_calls > 0 else 0,
        "completion_reasons": dict(completion_reasons),
    }


def evaluate_batch(
    input_path: str,
    output_path: Optional[str] = None,
    test_file: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate a JSONL file and keep full per-trial details.

    Supports:
    - Full per-trial details (preference-field diffs, utility breakdowns,
      validity issue lists, ...)
    - Grouping multiple trials by ``task_id`` while keeping both per-trial
      details and the averaged summary
    - Breakdown by ``difficulty_type`` (easy/medium/hard)
    - Tool-call statistics (cache_hits, cache_misses, tool_errors, ...)

    Args:
        input_path: Path to the inference-output JSONL.
        output_path: Optional path for the full detailed evaluation.
        test_file: Optional test-data path (used for ``difficulty_type``).

    Returns:
        Complete result dict with overall, grouped, and tool statistics.
    """
    # Load inference results
    results = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            result = json.loads(line)
            results.append(result)

    # Collect tool-call stats from the raw results
    tool_stats = _collect_tool_stats(results)

    # Evaluate each record (keep full details)
    eval_results = []
    for result in results:
        eval_result = evaluate_single_result(result)
        eval_results.append(eval_result)

    # Load the difficulty map
    difficulty_map = _load_difficulty_map(test_file)

    # Group by task_id, keeping every trial's full details
    detailed_results = _group_trials_with_details(eval_results, difficulty_map)

    # Group by task_id and average trials (for the summary)
    averaged_results = _average_trial_results(eval_results)

    # Attach difficulty_type to each averaged result
    for result in averaged_results:
        task_id = result.get("task_id", "")
        result["difficulty_type"] = difficulty_map.get(task_id, "unknown")

    # Global summary statistics
    overall_summary = _compute_summary(averaged_results)

    # Summary statistics by difficulty
    difficulty_groups: Dict[str, List[Dict]] = defaultdict(list)
    for result in averaged_results:
        difficulty_groups[result["difficulty_type"]].append(result)

    difficulty_summaries = {}
    for difficulty, group_results in sorted(difficulty_groups.items()):
        difficulty_summaries[difficulty] = {
            "count": len(group_results),
            "summary": _compute_summary(group_results),
        }

    output = {
        "input_file": input_path,
        "test_file": test_file,
        "total_raw_samples": len(results),
        "total_tasks": len(detailed_results),
        "tool_stats": tool_stats,
        "overall_summary": overall_summary,
        "difficulty_summaries": difficulty_summaries,
        "details": detailed_results,
    }

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

    return output

def _group_trials_with_details(
    eval_results: List[Dict[str, Any]],
    difficulty_map: Dict[str, str],
) -> List[Dict[str, Any]]:
    """
    Group by ``task_id``, keep full per-trial details, and average scores.

    Returns:
        One record per ``task_id`` containing:
        - trials: full evaluation details for each trial
        - averaged_scores: mean scores across trials
    """
    task_groups: Dict[str, List[Dict]] = defaultdict(list)
    for result in eval_results:
        task_id = result.get("task_id", "unknown")
        task_groups[task_id].append(result)

    detailed_results = []
    for task_id, trials in task_groups.items():
        task_entry = {
            "task_id": task_id,
            "difficulty_type": difficulty_map.get(task_id, "unknown"),
            "num_trials": len(trials),
            "plan_generated": any(t.get("plan_generated", False) for t in trials),
            "trials": trials,
            "averaged_scores": _compute_trial_averages(trials),
        }
        detailed_results.append(task_entry)

    return detailed_results

def _compute_trial_averages(trials: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Average scores across multiple trials."""
    averages = {}

    # preference_completeness
    pc_ratios = [
        t["preference_completeness"]["completeness_ratio"]
        for t in trials
        if t.get("preference_completeness") and "completeness_ratio" in t["preference_completeness"]
    ]
    if pc_ratios:
        averages["preference_completeness_ratio"] = round(sum(pc_ratios) / len(pc_ratios), 2)

    # group_utility
    gu_scores = [
        t["group_utility"]["total_score"]
        for t in trials
        if t.get("group_utility") and "total_score" in t["group_utility"]
    ]
    gu_per_user_scores = [
        t["group_utility"]["per_user_average_score"]
        for t in trials
        if t.get("group_utility") and "per_user_average_score" in t["group_utility"]
    ]
    if gu_scores:
        averages["group_utility_total_score"] = round(sum(gu_scores) / len(gu_scores), 2)
    if gu_per_user_scores:
        averages["group_utility_per_user_average_score"] = round(sum(gu_per_user_scores) / len(gu_per_user_scores), 2)

    # group_fairness
    gf_scores = [
        t["group_fairness"]["fairness_score"]
        for t in trials
        if t.get("group_fairness") and "fairness_score" in t["group_fairness"]
    ]
    if gf_scores:
        averages["group_fairness_score"] = round(sum(gf_scores) / len(gf_scores), 2)

    # plan_validity
    pv_issues = [
        t["plan_validity"]["total_issues"]
        for t in trials
        if t.get("plan_validity") and "total_issues" in t["plan_validity"]
    ]
    pv_valids = [
        t["plan_validity"]["valid"]
        for t in trials
        if t.get("plan_validity") and "valid" in t["plan_validity"]
    ]
    if pv_valids:
        averages["plan_validity_valid_ratio"] = round(sum(pv_valids) / len(pv_valids), 2)
    if pv_issues:
        averages["plan_validity_avg_issues"] = round(sum(pv_issues) / len(pv_issues), 2)

    return averages


def _compute_summary(eval_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute summary statistics.

    For samples that never produced a plan (error state):
    - preference_completeness: scored as usual (does not depend on a plan)
    - group_utility / group_fairness / plan_validity: counted as 0 in the
      average (an unfinished task must not be dropped from the stats)
    """
    total = len(eval_results)
    if total == 0:
        return {}

    # Preference completeness (every sample is scorable; use the regular average)
    completeness_ratios = []
    for r in eval_results:
        pc = r.get("preference_completeness")
        if pc and "completeness_ratio" in pc:
            completeness_ratios.append(pc["completeness_ratio"])
        else:
            completeness_ratios.append(0.0)

    # Group utility (per person): use the score when present, else 0 for error
    utility_scores = []
    utility_valid_count = 0
    for r in eval_results:
        gu = r.get("group_utility")
        if gu and "per_user_average_score" in gu:
            utility_scores.append(gu["per_user_average_score"])
            utility_valid_count += 1
        elif gu and "total_score" in gu:
            utility_scores.append(gu["total_score"])
            utility_valid_count += 1
        else:
            utility_scores.append(0.0)

    # Group fairness: use the score when present, else 0 for error
    fairness_scores = []
    fairness_valid_count = 0
    for r in eval_results:
        gf = r.get("group_fairness")
        if gf and "fairness_score" in gf:
            fairness_scores.append(gf["fairness_score"])
            fairness_valid_count += 1
        else:
            fairness_scores.append(0.0)

    # Plan validity: average the per-trial valid ratio
    # Equivalent to valid_samples / total_samples (e.g. 100 tasks × 3 trials = 300 samples)
    validity_ratios = []
    for r in eval_results:
        pv = r.get("plan_validity")
        if pv and "valid_ratio_across_trials" in pv:
            # After trial averaging, use the between-trial valid ratio
            validity_ratios.append(pv["valid_ratio_across_trials"])
        elif pv and "valid" in pv:
            # Single-trial record: valid is a boolean
            validity_ratios.append(1.0 if pv["valid"] else 0.0)
        else:
            # error (e.g. no plan generated) counts as invalid
            validity_ratios.append(0.0)

    summary = {
        "preference_completeness": {
            "count": total,
            "average_ratio": round(sum(completeness_ratios) / total, 2) if total > 0 else 0,
            "min_ratio": round(min(completeness_ratios), 2) if completeness_ratios else 0,
            "max_ratio": round(max(completeness_ratios), 2) if completeness_ratios else 0,
        },
        "group_utility": {
            "count": total,
            "valid_count": utility_valid_count,
            "average_score": round(sum(utility_scores) / total, 2) if total > 0 else 0,
            "min_score": round(min(utility_scores), 2) if utility_scores else 0,
            "max_score": round(max(utility_scores), 2) if utility_scores else 0,
        },
        "group_fairness": {
            "count": total,
            "valid_count": fairness_valid_count,
            "average_fairness_score": round(sum(fairness_scores) / total, 2) if total > 0 else 0,
            "min_fairness_score": round(min(fairness_scores), 2) if fairness_scores else 0,
            "max_fairness_score": round(max(fairness_scores), 2) if fairness_scores else 0,
        },
        "plan_validity": {
            "count": total,
            "valid_ratio": round(sum(validity_ratios) / total, 2) if total > 0 else 0,
        },
    }

    return summary


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Multi-user travel-planning evaluation")
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input JSONL path (inference output)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output JSON path (optional; stdout if omitted)",
    )
    parser.add_argument(
        "--test-file", "-t",
        default=None,
        help="Test-data path (used for difficulty_type)",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print summary statistics only, without per-task details",
    )
    args = parser.parse_args()

    output = evaluate_batch(args.input, args.output, test_file=args.test_file)

    if args.summary_only:
        print(json.dumps(output["overall_summary"], ensure_ascii=False, indent=2))
    elif not args.output:
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        print(f"评测完成，结果已保存到: {args.output}")
        print(f"\n总任务数: {output['total_tasks']} (原始样本: {output['total_raw_samples']})")
        print(f"\n=== Overall Summary ===")
        print(json.dumps(output["overall_summary"], ensure_ascii=False, indent=2))
        if output.get("difficulty_summaries"):
            print(f"\n=== By Difficulty ===")
            for diff, info in output["difficulty_summaries"].items():
                print(f"\n--- {diff} ({info['count']} tasks) ---")
                print(json.dumps(info["summary"], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
