#!/usr/bin/env python3
"""
LLM-Judge evaluation.

Score a multi-user travel-planning Agent's conversation along five
dimensions with a large language model. Supports concurrent judging
and an optional meta-judge review pass.

Usage:
    python -m group_travelbench.evaluation.llm_judge \
        --input /path/to/result.jsonl \
        --output /path/to/judge_output.json \
        --api_key YOUR_KEY \
        --model gpt-4o
"""

import json
import argparse
import time
import os
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from pathlib import Path

from openai import OpenAI

from .llm_judge_prompt import (
    SYSTEM_PROMPT,
    MULTI_AGENT_JUDGE_PROMPT,
    META_JUDGE_SYSTEM_PROMPT,
    META_JUDGE_PROMPT,
    DIMENSIONS,
)
from ..core.thinking import build_thinking_extra_body, drop_unsupported_params
from ..utils.eval_util import fix_xml_tags, ResultParser


# =============================================================================
# Conversation History Formatter
# =============================================================================

def format_conversation_for_judge(conversation_history: List[Dict]) -> str:
    """
    Format ``conversation_history`` as text for the LLM judge.

    Includes user messages, Agent replies, tool-call arguments, and tool
    results.

    Args:
        conversation_history: Conversation-history list.

    Returns:
        Formatted conversation text.
    """
    lines = []

    for msg in conversation_history:
        msg_type = msg.get("type", "")

        if msg_type == "message":
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            has_tool_calls = msg.get("has_tool_calls", False)

            if role == "Agent":
                if content:
                    lines.append(f"[Agent]: {content}")
                if has_tool_calls and not content:
                    lines.append("[Agent]: (调用工具，见下方)")
            else:
                lines.append(f"[{role}]: {content}")

        elif msg_type == "tool_call":
            tool_name = msg.get("name", "unknown")
            arguments = msg.get("arguments", "")
            if isinstance(arguments, dict):
                arguments = json.dumps(arguments, ensure_ascii=False)
            lines.append(f"  [工具调用] {tool_name}")
            lines.append(f"  参数: {arguments}")

        elif msg_type == "tool_result":
            tool_name = msg.get("name", "unknown")
            result = msg.get("result", "")
            error = msg.get("error", False)
            # Truncate long tool returns (stay inside the context window)
            if len(result) > 2000:
                result = result[:2000] + "\n  ... (结果已截断)"
            if error:
                lines.append(f"  [工具返回-错误] {tool_name}: {result}")
            else:
                lines.append(f"  [工具返回] {tool_name}:")
                lines.append(f"  {result}")

        lines.append("")  # Blank-line separator

    return "\n".join(lines)


# =============================================================================
# Configuration
# =============================================================================

class JudgeConfig:
    """Configuration for LLM-Judge evaluation.

    The judge must be a strong *thinking* model: it reads whole multi-user
    conversations and has to reason about preference coverage, fairness and plan
    validity. ``enable_thinking`` defaults to True and is sent to the backend as
    an explicit switch (this work uses Gemini3-Flash-Preview).
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o",
        temperature: float = 0.1,
        max_tokens: int = 32768,
        max_retries: int = 5,
        enable_thinking: bool = True,
        enable_meta_judge: bool = False,
        meta_judge_model: Optional[str] = None,
        meta_judge_api_key: Optional[str] = None,
        meta_judge_base_url: Optional[str] = None,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.enable_thinking = enable_thinking
        self.enable_meta_judge = enable_meta_judge
        self.meta_judge_model = meta_judge_model or model
        self.meta_judge_api_key = meta_judge_api_key or api_key
        self.meta_judge_base_url = meta_judge_base_url or base_url
        self.dimensions = DIMENSIONS


# =============================================================================
# LLM Client
# =============================================================================

class JudgeLLMClient:
    """Thin wrapper around the LLM client."""

    def __init__(self, config: JudgeConfig):
        self.config = config
        self.client = OpenAI(api_key=config.api_key, base_url=config.base_url)
        self.lock = Lock()

    def call(
        self,
        messages: List[Dict],
        system_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        client: Optional[OpenAI] = None,
    ) -> Optional[str]:
        """Call the LLM and return the text response."""
        temperature = temperature if temperature is not None else self.config.temperature
        max_tokens = max_tokens or self.config.max_tokens
        model = model or self.config.model
        client = client or self.client

        final_messages = [{"role": "system", "content": system_prompt}] + messages

        # Explicit thinking switch, same convention as the agent / simulator
        # clients: the judge is expected to be a strong thinking model.
        request_params: Dict = {
            "messages": final_messages,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        extra_body = build_thinking_extra_body(
            enable_thinking=self.config.enable_thinking,
        )
        if extra_body:
            request_params["extra_body"] = extra_body

        attempt = 0

        while attempt < self.config.max_retries:
            try:
                response = client.chat.completions.create(**request_params)
                content = response.choices[0].message.content
                if content is not None:
                    return content
                with self.lock:
                    print(f"[Warning] Attempt {attempt + 1}/{self.config.max_retries}: Response content is None")

            except Exception as e:
                error_name = type(e).__name__
                with self.lock:
                    print(f"[Warning] Attempt {attempt + 1}/{self.config.max_retries} failed: {error_name}: {e}")
                # Some gateways reject the optional thinking keys with a 4xx: drop
                # them and retry right away, without consuming the retry budget.
                # The endpoint key is namespaced so that a rejection seen here
                # cannot disable thinking for the agent / simulators, which share
                # OPENAI_API_BASE but send different payloads.
                if drop_unsupported_params(
                    request_params, e, f"llm-judge:{self.config.base_url}"
                ):
                    continue
                # APIConnectionError cannot be resolved by retrying, raise directly
                if error_name == "APIConnectionError":
                    with self.lock:
                        print("[Error] APIConnectionError detected. Connection error cannot be resolved by retrying.")
                    raise

            attempt += 1
            time.sleep(1)

        with self.lock:
            print("[Error] Maximum retry attempts reached. No content retrieved.")
        return None


# =============================================================================
# Judge Evaluator
# =============================================================================

class MultiAgentJudge:
    """LLM-Judge evaluator for a multi-user travel-planning Agent."""

    def __init__(self, config: JudgeConfig):
        self.config = config
        self.llm_client = JudgeLLMClient(config)
        self.result_parser = ResultParser()

        if config.enable_meta_judge:
            self.meta_judge_client = OpenAI(
                api_key=config.meta_judge_api_key,
                base_url=config.meta_judge_base_url,
            )
        else:
            self.meta_judge_client = None

    def evaluate_single(self, result: Dict, index: int) -> Optional[Dict]:
        """Evaluate a single result."""
        task_id = result.get("task_id", "unknown")
        trial_id = result.get("trial_id", 0)
        query = result.get("query", "")
        context = result.get("context", "")
        user_names = result.get("user_names", [])
        conversation_history = result.get("conversation_history", [])
        original_preferences = result.get("original_preferences", {})

        # Inference-crash sample: an empty conversation_history means the Agent-side
        # run failed (e.g. context_overflow / rate_limit / connection_error /
        # runtime_error) and produced no dialogue at all. This is a capability
        # failure of the evaluated Agent and must count as 0 (same convention as
        # multi_eval for crashed samples), not be dropped — otherwise models that
        # crash more often would get an inflated LLM-Judge average via survivorship bias.
        #
        # Note: this is distinct from an LLM-Judge call/parse failure (the parsed_result
        # is-None branch below). That is a framework issue, unrelated to the Agent,
        # and still returns None so it is discarded and does not bias the score.
        if not conversation_history:
            with self.llm_client.lock:
                print(f"[{index}] task={task_id} trial={trial_id} Inference crashed (empty conversation_history): scored 0")
            zero_scores = {f"{dim}_score": 1 for dim in self.config.dimensions}
            zero_scores["average_score"] = 1
            zero_normalized = {f"{dim}_score": 0.0 for dim in self.config.dimensions}
            zero_normalized["average_score"] = 0.0
            return {
                "index": index,
                "task_id": task_id,
                "trial_id": trial_id,
                "query": query,
                "inference_crashed": True,
                "evaluation": {
                    "raw_response": None,
                    "parsed_result": None,
                    "scores": zero_scores,
                    "normalized_scores": zero_normalized,
                    "meta_judge_result": None,
                    "final_scores": dict(zero_normalized),
                    "evaluation_time": 0.0,
                },
            }

        # Format the conversation trace
        formatted_conversation = format_conversation_for_judge(conversation_history)

        # Format the original preference table (pre-compromise; used for process quality)
        formatted_preferences = json.dumps(
            original_preferences, ensure_ascii=False, indent=2
        ) if original_preferences else "（无参考偏好数据）"

        # Build the prompt
        evaluation_prompt = MULTI_AGENT_JUDGE_PROMPT.format(
            QUERY=query,
            CONTEXT=context,
            USER_NAMES=", ".join(user_names),
            EFFECTIVE_PREFERENCES=formatted_preferences,
            CONVERSATION_HISTORY=formatted_conversation,
        )

        # Call the LLM for judging (with parse retries)
        start_time = time.time()
        parsed_result, scores, raw_response = self._call_and_parse(evaluation_prompt)
        elapsed_time = time.time() - start_time

        if not parsed_result or not scores:
            with self.llm_client.lock:
                print(f"[{index}] task={task_id} trial={trial_id} Evaluation failed: unable to get valid score after retries")
            return None

        # Normalize scores to 0-100
        normalized_scores = {}
        for dim in self.config.dimensions:
            score_key = f"{dim}_score"
            raw_score = scores.get(score_key)
            if raw_score is not None:
                normalized_scores[score_key] = round((raw_score - 1) / 4 * 100, 2)
            else:
                normalized_scores[score_key] = None
        normalized_scores["average_score"] = round((scores["average_score"] - 1) / 4 * 100, 2)

        # Meta-judge (if enabled)
        meta_judge_result = None
        final_scores = normalized_scores.copy()

        if self.config.enable_meta_judge and self.meta_judge_client is not None:
            meta_judge_result = self._meta_judge(
                query, context, user_names,
                formatted_conversation, formatted_preferences,
                scores, parsed_result,
            )
            if meta_judge_result and meta_judge_result.get("meta_score") is not None:
                meta_weight = meta_judge_result["meta_score"] / 5.0
                final_scores = {}
                for dim in self.config.dimensions:
                    score_key = f"{dim}_score"
                    if normalized_scores.get(score_key) is not None:
                        final_scores[score_key] = round(normalized_scores[score_key] * meta_weight, 2)
                    else:
                        final_scores[score_key] = None
                final_scores["average_score"] = round(normalized_scores["average_score"] * meta_weight, 2)

        eval_result = {
            "index": index,
            "task_id": task_id,
            "trial_id": trial_id,
            "query": query,
            "evaluation": {
                "raw_response": raw_response,
                "parsed_result": parsed_result,
                "scores": scores,
                "normalized_scores": normalized_scores,
                "meta_judge_result": meta_judge_result,
                "final_scores": final_scores,
                "evaluation_time": elapsed_time,
            },
        }

        with self.llm_client.lock:
            avg = final_scores.get("average_score", 0) or 0
            raw_avg = scores.get("average_score", 0) or 0
            if self.config.enable_meta_judge and meta_judge_result and meta_judge_result.get("meta_score") is not None:
                print(f"[{index}] task={task_id} trial={trial_id}, raw avg: {raw_avg:.2f}, meta-judge: {meta_judge_result['meta_score']}, final avg (0-100): {avg:.2f}, time: {elapsed_time:.1f}s")
            else:
                print(f"[{index}] task={task_id} trial={trial_id}, raw avg: {raw_avg:.2f}, final avg (0-100): {avg:.2f}, time: {elapsed_time:.1f}s")

        return eval_result

    def _call_and_parse(
        self, prompt: str
    ) -> tuple:
        """Call the LLM and parse the XML response, with retries."""
        messages = [{"role": "user", "content": prompt}]
        max_parse_retries = 10
        for attempt in range(max_parse_retries):
            temperature = 0.7 if attempt > 6 else self.config.temperature
            try:
                response = self.llm_client.call(
                    messages=messages,
                    system_prompt=SYSTEM_PROMPT,
                    temperature=temperature,
                )
                if not response:
                    with self.llm_client.lock:
                        print(f"  [Parse retry {attempt + 1}/{max_parse_retries}] LLM returned empty response")
                    continue
            except Exception as e:
                error_name = type(e).__name__
                with self.llm_client.lock:
                    print(f"  [Parse retry {attempt + 1}/{max_parse_retries}] LLM call exception: {error_name}: {e}")
                # APIConnectionError cannot be resolved by retrying, abort immediately
                if error_name == "APIConnectionError":
                    return None, None, None
                continue

            # Try parsing XML
            parsed_result = self.result_parser.parse_xml_response(
                response, self.config.dimensions
            )
            if not parsed_result:
                with self.llm_client.lock:
                    # Keep the first 200 characters of the response for diagnostics
                    preview = response[:200].replace('\n', ' ')
                    print(f"  [Parse retry {attempt + 1}/{max_parse_retries}] XML parse failed. Response preview: {preview}...")
            elif parsed_result:
                scores = self.result_parser.extract_scores(
                    parsed_result, self.config.dimensions
                )
                if scores and scores.get("average_score") is not None:
                    return parsed_result, scores, response
                else:
                    with self.llm_client.lock:
                        print(f"  [Parse retry {attempt + 1}/{max_parse_retries}] Score extraction failed. Parsed dimensions: {list(parsed_result.keys())}, extracted scores: {scores}")

            if attempt < max_parse_retries - 1:
                time.sleep(1)

        with self.llm_client.lock:
            print(f"  [Parse] All {max_parse_retries} retries exhausted, giving up.")
        return None, None, None

    def _meta_judge(
        self,
        query: str,
        context: str,
        user_names: List[str],
        formatted_conversation: str,
        formatted_preferences: str,
        scores: Dict,
        parsed_result: Dict,
    ) -> Optional[Dict]:
        """Run the meta-judge review."""
        # Build the score summary
        score_lines = []
        for dim in self.config.dimensions:
            score_key = f"{dim}_score"
            score = scores.get(score_key, "N/A")
            score_lines.append(f"- {dim}: {score}")
        score_lines.append(f"- average: {scores.get('average_score', 'N/A')}")
        evaluation_scores = "\n".join(score_lines)

        # Build the reasoning summary
        reasoning_lines = []
        for dim in self.config.dimensions:
            dim_result = parsed_result.get(dim, {})
            reasoning = dim_result.get("reasoning", "无推理内容")
            rating = dim_result.get("rating", "无评级")
            reasoning_lines.append(f"### {dim}")
            reasoning_lines.append(f"**评级**: {rating}")
            reasoning_lines.append(f"**推理**: {reasoning}")
            reasoning_lines.append("")
        evaluation_reasoning = "\n".join(reasoning_lines)

        meta_prompt = META_JUDGE_PROMPT.format(
            QUERY=query,
            CONTEXT=context,
            USER_NAMES=", ".join(user_names),
            EFFECTIVE_PREFERENCES=formatted_preferences,
            CONVERSATION_HISTORY=formatted_conversation,
            EVALUATION_SCORES=evaluation_scores,
            EVALUATION_REASONING=evaluation_reasoning,
        )

        messages = [{"role": "user", "content": meta_prompt}]
        max_parse_retries = 10

        for attempt in range(max_parse_retries):
            try:
                response = self.llm_client.call(
                    messages=messages,
                    system_prompt=META_JUDGE_SYSTEM_PROMPT,
                    model=self.config.meta_judge_model,
                    client=self.meta_judge_client,
                )
                if not response:
                    continue
            except Exception as e:
                error_name = type(e).__name__
                if error_name == "APIConnectionError":
                    return None
                continue

            parsed_meta = self.result_parser.parse_meta_judge_response(response)
            if parsed_meta:
                rating = parsed_meta.get("rating", "")
                meta_score = self.result_parser.rating_to_score(rating)
                if meta_score is not None:
                    return {
                        "raw_response": response,
                        "parsed_result": parsed_meta,
                        "meta_score": meta_score,
                        "meta_rating": rating,
                    }

            if attempt < max_parse_retries - 1:
                time.sleep(1)

        return None

    def evaluate_batch(
        self, results: List[Dict], num_workers: int = 4
    ) -> List[Dict]:
        """Evaluate a batch of results."""
        eval_results = []

        if num_workers == 1:
            for i, result in enumerate(results):
                eval_result = self.evaluate_single(result, i)
                if eval_result is not None:
                    eval_results.append(eval_result)
        else:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                future_to_index = {}

                # Submit in batches to stay under QPS limits
                batch_size = 30
                for batch_start in range(0, len(results), batch_size):
                    batch_end = min(batch_start + batch_size, len(results))
                    for i in range(batch_start, batch_end):
                        future = executor.submit(
                            self.evaluate_single, results[i], i
                        )
                        future_to_index[future] = i

                    if batch_end < len(results):
                        time.sleep(10)

                for future in as_completed(future_to_index):
                    try:
                        eval_result = future.result()
                        if eval_result is not None:
                            eval_results.append(eval_result)
                    except Exception as e:
                        idx = future_to_index[future]
                        print(f"[Error] Error evaluating trajectory {idx}: {type(e).__name__}: {e}")
                        import traceback
                        traceback.print_exc()

        eval_results.sort(key=lambda x: x["index"])
        return eval_results


# =============================================================================
# Statistics
# =============================================================================

def compute_judge_statistics(eval_results: List[Dict]) -> Dict:
    """Compute aggregate statistics for LLM-Judge results."""
    if not eval_results:
        return {}

    dimension_scores = {dim: [] for dim in DIMENSIONS}
    average_scores = []
    final_average_scores = []
    meta_scores = []

    for r in eval_results:
        evaluation = r.get("evaluation", {})
        scores = evaluation.get("scores", {})
        final_scores = evaluation.get("final_scores", {})

        for dim in DIMENSIONS:
            score_key = f"{dim}_score"
            if scores.get(score_key) is not None:
                dimension_scores[dim].append(scores[score_key])

        if scores.get("average_score") is not None:
            average_scores.append(scores["average_score"])

        if final_scores.get("average_score") is not None:
            final_average_scores.append(final_scores["average_score"])

        meta_result = evaluation.get("meta_judge_result")
        if meta_result and meta_result.get("meta_score") is not None:
            meta_scores.append(meta_result["meta_score"])

    statistics = {
        "total_evaluated": len(eval_results),
        "dimensions": DIMENSIONS,
        "raw_scores_1_5": {
            dim: {
                "mean": round(sum(scores) / len(scores), 2) if scores else 0,
                "min": min(scores) if scores else 0,
                "max": max(scores) if scores else 0,
                "count": len(scores),
            }
            for dim, scores in dimension_scores.items()
        },
        "raw_average_1_5": round(
            sum(average_scores) / len(average_scores), 2
        ) if average_scores else 0,
        "final_average_0_100": round(
            sum(final_average_scores) / len(final_average_scores), 2
        ) if final_average_scores else 0,
    }

    if meta_scores:
        statistics["meta_judge"] = {
            "count": len(meta_scores),
            "mean": round(sum(meta_scores) / len(meta_scores), 2),
            "distribution": {i: meta_scores.count(i) for i in range(1, 6)},
        }

    return statistics


# =============================================================================
# Main Entry
# =============================================================================

def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="LLM-Judge for multi-user travel planning")
    parser.add_argument("--input", "-i", required=True, help="Input JSONL path")
    parser.add_argument("--output", "-o", required=True, help="Output JSON path")
    parser.add_argument("--api_key", default=None, help="OpenAI API Key")
    parser.add_argument(
        "--base_url", default="https://api.openai.com/v1", help="API Base URL"
    )
    parser.add_argument("--model", default="gpt-4o", help="Judge model")
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Send an explicit thinking OFF switch to the judge backend "
             "(default: thinking ON — the judge should be a strong thinking model)",
    )
    parser.add_argument("--temperature", type=float, default=0.1, help="Temperature")
    parser.add_argument("--max_tokens", type=int, default=16384, help="Max tokens")
    parser.add_argument(
        "--max-concurrency", type=int, default=4, help="Maximum concurrency"
    )
    parser.add_argument(
        "--max_samples", type=int, default=None, help="Maximum number of samples to judge"
    )
    parser.add_argument(
        "--enable_meta_judge", action="store_true", help="Enable meta-judge"
    )
    parser.add_argument(
        "--meta_judge_model", default=None, help="Meta-judge 模型"
    )
    parser.add_argument(
        "--meta_judge_api_key", default=None, help="Meta-judge API Key"
    )
    parser.add_argument(
        "--meta_judge_base_url", default=None, help="Meta-judge Base URL"
    )
    args = parser.parse_args()

    # Resolve the API key
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[Error] API Key 未配置，请通过 --api_key 或 OPENAI_API_KEY 环境变量设置")
        return

    base_url = args.base_url
    if base_url == "https://api.openai.com/v1":
        env_base_url = os.getenv("OPENAI_API_BASE")
        if env_base_url:
            base_url = env_base_url

    config = JudgeConfig(
        api_key=api_key,
        base_url=base_url,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        enable_thinking=not args.disable_thinking,
        enable_meta_judge=args.enable_meta_judge,
        meta_judge_model=args.meta_judge_model,
        meta_judge_api_key=args.meta_judge_api_key,
        meta_judge_base_url=args.meta_judge_base_url,
    )

    print("=" * 60)
    print("多人旅行规划 LLM-Judge 评测")
    print("=" * 60)
    print(f"输入文件: {args.input}")
    print(f"输出文件: {args.output}")
    print(f"模型: {args.model}")
    print(f"评测维度: {', '.join(DIMENSIONS)}")
    print(f"并发数: {args.max_concurrency}")
    print(f"Meta-judge: {'启用' if args.enable_meta_judge else '禁用'}")
    if args.enable_meta_judge:
        print(f"Meta-judge 模型: {config.meta_judge_model}")
    print("=" * 60)

    # Load data
    print("\n[1/3] 加载数据...")
    results = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))

    if args.max_samples:
        results = results[: args.max_samples]
        print(f"限制为 {args.max_samples} 条样本")

    print(f"共 {len(results)} 条待评测")

    # Evaluate
    print(f"\n[2/3] 开始评测...")
    judge = MultiAgentJudge(config)
    start_time = time.time()
    eval_results = judge.evaluate_batch(results, num_workers=args.max_concurrency)
    total_time = time.time() - start_time

    print(f"\n评测完成，总耗时: {total_time:.2f}s")
    print(f"成功评测: {len(eval_results)}/{len(results)}")

    # Statistics
    print("\n[3/3] 计算统计...")
    statistics = compute_judge_statistics(eval_results)

    print(f"\n{'='*40}")
    print("评测结果汇总")
    print(f"{'='*40}")
    print(f"各维度原始分 (1-5):")
    for dim in DIMENSIONS:
        dim_stats = statistics.get("raw_scores_1_5", {}).get(dim, {})
        print(f"  {dim}: {dim_stats.get('mean', 0):.2f}")
    print(f"  平均: {statistics.get('raw_average_1_5', 0):.2f}")
    print(f"\n最终平均分 (0-100): {statistics.get('final_average_0_100', 0):.2f}")

    if "meta_judge" in statistics:
        meta = statistics["meta_judge"]
        print(f"\nMeta-judge: mean={meta['mean']:.2f}, count={meta['count']}")

    # Save results
    output = {
        "input_file": args.input,
        "config": {
            "model": args.model,
            "temperature": args.temperature,
            "enable_meta_judge": args.enable_meta_judge,
            "meta_judge_model": config.meta_judge_model,
        },
        "statistics": statistics,
        "details": eval_results,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存到: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
