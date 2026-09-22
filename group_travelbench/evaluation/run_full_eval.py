"""
Full evaluation pipeline.

Chain multi_eval (rule-based) + llm_judge (LLM-Judge) +
report_generator so inference output becomes a final report in one
pass.

Usage:
    python -m group_travelbench.evaluation.run_full_eval \
        --input infer_output/Agent-xxx/test.jsonl \
        --output-dir eval_output/Agent-xxx \
        --test-file datas/test.jsonl \
        --agent-name "Agent-your-model-name"
"""

import json
import os
import argparse
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .multi_eval import evaluate_batch
from .llm_judge import MultiAgentJudge, JudgeConfig, compute_judge_statistics
from .report_generator import generate_report


def run_full_eval(
    input_path: str,
    output_dir: str,
    test_file: str,
    agent_name: str,
    llm_judge_model: str = "gpt-4o",
    llm_judge_concurrency: int = 30,
    enable_meta_judge: bool = True,
    skip_llm_judge: bool = False,
    disable_thinking: bool = False,
) -> None:
    """
    Run the full evaluation pipeline.

    Steps:
        1. multi_eval: rule-based scoring (completeness, utility, fairness, validity)
        2. llm_judge: LLM-Judge scoring (multi-dimension + meta-judge)
        3. report: generate the Markdown report
    """
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    multi_eval_output_path = str(output_dir_path / "multi_eval_result.json")
    llm_judge_output_path = str(output_dir_path / "llm_judge_result.json")
    report_output_path = str(output_dir_path / "report.md")

    # ==================================================================
    # Step 1: Rule-based Evaluation (multi_eval)
    # ==================================================================
    print("=" * 60)
    print("[1/3] Running Rule-based Evaluation (multi_eval)")
    print("=" * 60)
    print(f"  Input: {input_path}")
    print(f"  Test file: {test_file}")
    print()

    start_time = time.time()
    multi_eval_output = evaluate_batch(
        input_path=input_path,
        output_path=multi_eval_output_path,
        test_file=test_file,
    )
    elapsed = time.time() - start_time

    print(f"  ✅ multi_eval completed in {elapsed:.1f}s")
    print(f"  Total tasks: {multi_eval_output['total_tasks']}")
    print(f"  Output: {multi_eval_output_path}")
    print()

    # ==================================================================
    # Step 2: LLM-Judge Evaluation
    # ==================================================================
    llm_judge_output = None

    if skip_llm_judge:
        print("=" * 60)
        print("[2/3] LLM-Judge Evaluation (SKIPPED)")
        print("=" * 60)
        print()

        # Check if a previous result exists
        if Path(llm_judge_output_path).exists():
            print(f"  Loading existing LLM-Judge result: {llm_judge_output_path}")
            with open(llm_judge_output_path, "r", encoding="utf-8") as f:
                llm_judge_output = json.load(f)
            print()
    else:
        print("=" * 60)
        print("[2/3] Running LLM-Judge Evaluation")
        print("=" * 60)

        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")

        if not api_key:
            print("  ⚠️  OPENAI_API_KEY not set, skipping LLM-Judge")
            print()
        else:
            print(f"  Model: {llm_judge_model}")
            print(f"  Concurrency: {llm_judge_concurrency}")
            print(f"  Thinking: {'disabled' if disable_thinking else 'enabled'}")
            print(f"  Meta-Judge: {'enabled' if enable_meta_judge else 'disabled'}")
            print()

            config = JudgeConfig(
                api_key=api_key,
                base_url=base_url,
                model=llm_judge_model,
                temperature=0.1,
                max_tokens=32768,
                enable_thinking=not disable_thinking,
                enable_meta_judge=enable_meta_judge,
                meta_judge_model=llm_judge_model,
                meta_judge_api_key=api_key,
                meta_judge_base_url=base_url,
            )

            # Load ALL inference results for LLM-Judge (all trials).
            # Each trial has a different conversation, so each should be judged.
            # Results are later averaged by task_id in the report generator.
            all_results = []
            with open(input_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        all_results.append(json.loads(line))

            print(f"  Evaluating {len(all_results)} samples (all trials)...")
            print()

            start_time = time.time()
            judge = MultiAgentJudge(config)
            eval_results = judge.evaluate_batch(
                all_results, num_workers=llm_judge_concurrency
            )
            elapsed = time.time() - start_time

            statistics = compute_judge_statistics(eval_results)

            llm_judge_output = {
                "input_file": input_path,
                "config": {
                    "model": llm_judge_model,
                    "temperature": 0,
                    "enable_thinking": not disable_thinking,
                    "enable_meta_judge": enable_meta_judge,
                    "meta_judge_model": llm_judge_model,
                },
                "statistics": statistics,
                "details": eval_results,
            }

            with open(llm_judge_output_path, "w", encoding="utf-8") as f:
                json.dump(llm_judge_output, f, ensure_ascii=False, indent=2)

            print(f"  ✅ LLM-Judge completed in {elapsed:.1f}s")
            print(f"  Successfully evaluated: {len(eval_results)}/{len(all_results)}")
            print(f"  Final average (0-100): {statistics.get('final_average_0_100', 0):.2f}")
            print(f"  Output: {llm_judge_output_path}")
            print()

    # ==================================================================
    # Step 3: Generate Report
    # ==================================================================
    print("=" * 60)
    print("[3/3] Generating Evaluation Report")
    print("=" * 60)

    report_text = generate_report(
        multi_eval_output=multi_eval_output,
        llm_judge_output=llm_judge_output,
        agent_name=agent_name,
        output_path=report_output_path,
    )

    print(f"  ✅ Report generated: {report_output_path}")
    print()
    print("=" * 60)
    print("🎉 Full evaluation pipeline completed!")
    print("=" * 60)
    print(f"  📄 Rule-based eval: {multi_eval_output_path}")
    if llm_judge_output:
        print(f"  📄 LLM-Judge eval:  {llm_judge_output_path}")
    print(f"  📊 Report:          {report_output_path}")
    print()

    # Print a brief summary to console
    print("=" * 60)
    print("📊 Quick Summary")
    print("=" * 60)
    print(report_text[:2000])
    if len(report_text) > 2000:
        print(f"\n... (full report at {report_output_path})")


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="GroupTravelbench full evaluation pipeline")
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to the inference-output JSONL",
    )
    parser.add_argument(
        "--output-dir", "-o", required=True,
        help="Directory for evaluation outputs",
    )
    parser.add_argument(
        "--test-file", "-t", required=True,
        help="Test-data path (used for difficulty_type)",
    )
    parser.add_argument(
        "--agent-name", required=True,
        help="Agent name (report title and directory naming)",
    )
    parser.add_argument(
        "--llm-judge-model", default="gpt-4o",
        help="Model used by LLM-Judge (default: gpt-4o)",
    )
    parser.add_argument(
        "--llm-judge-concurrency", type=int, default=30,
        help="LLM-Judge concurrency (default: 30)",
    )
    parser.add_argument(
        "--enable-meta-judge", action="store_true", default=True,
        help="Enable meta-judge (on by default)",
    )
    parser.add_argument(
        "--no-meta-judge", action="store_true",
        help="Disable meta-judge",
    )
    parser.add_argument(
        "--skip-llm-judge", action="store_true",
        help="Skip LLM-Judge (rule-based evaluation only)",
    )
    parser.add_argument(
        "--disable-thinking", action="store_true",
        help="Send an explicit thinking OFF switch to the judge backend "
             "(default: thinking ON — the judge should be a strong thinking model)",
    )
    args = parser.parse_args()

    enable_meta_judge = args.enable_meta_judge and not args.no_meta_judge

    run_full_eval(
        input_path=args.input,
        output_dir=args.output_dir,
        test_file=args.test_file,
        agent_name=args.agent_name,
        llm_judge_model=args.llm_judge_model,
        llm_judge_concurrency=args.llm_judge_concurrency,
        enable_meta_judge=enable_meta_judge,
        skip_llm_judge=args.skip_llm_judge,
        disable_thinking=args.disable_thinking,
    )


if __name__ == "__main__":
    main()
