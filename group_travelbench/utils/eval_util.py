"""
Utility functions and classes for trajectory evaluation.

This module provides reusable components for the evaluation pipeline:
- XML parsing and fixing utilities
- Prompt building utilities
- Result parsing utilities
- Data loading utilities
- Statistics calculation utilities
"""

import json
import re
from typing import List, Dict, Optional
import xml.etree.ElementTree as ET

__all__ = [
    'fix_xml_tags',
    'PromptBuilder',
    'ResultParser',
    'DataLoader',
    'StatisticsCalculator',
    'POICategoryResolver',
    'POIOpeningHoursResolver',
]


# ==================== XML Utilities ====================

def fix_xml_tags(xml_content: str, dimensions: List[str]) -> str:
    """
    Attempt to repair common XML tag issues in LLM-Judge evaluation responses.

    Handles the most frequent LLM output problems:
    1. Missing or extra <response> wrapper
    2. XML-unsafe characters inside <reasoning> text (< > & etc.)
    3. Missing closing tags (</reasoning>, </rating>, </dimension>)
    4. Extra whitespace or slight variations in tag names
    5. HTML-like tags inside reasoning (e.g. <br>, <b>, <!-- -->)

    Args:
        xml_content: XML content string to fix.
        dimensions: List of dimension names (e.g. ["hallucination_factuality", ...]).

    Returns:
        A best-effort repaired XML string. May still fail for severely malformed input.
    """
    # -------------------------------------------------------------------------
    # Strategy: use regex-based extraction per dimension instead of full XML
    # parsing. This is far more robust against malformed content inside
    # <reasoning> blocks (which is the #1 failure mode).
    # -------------------------------------------------------------------------

    # Step 1: Normalize whitespace in tag names (LLM sometimes adds spaces)
    # e.g. "< reasoning >" -> "<reasoning>"
    xml_content = re.sub(r'<\s+', '<', xml_content)
    xml_content = re.sub(r'\s+>', '>', xml_content)
    xml_content = re.sub(r'<\s*/\s*', '</', xml_content)

    # Step 2: Try to fix tag name typos by fuzzy matching dimension names
    # e.g. "hallucination_factuality" might be written as "hallucination"
    all_valid_tags = set(dimensions + ["response", "reasoning", "rating",
                                        "meta_evaluation"])
    tag_pattern = re.compile(r'</?([a-zA-Z_][a-zA-Z0-9_]*)>')

    def _fix_tag_name(match):
        tag_full = match.group(0)
        tag_name = match.group(1)
        if tag_name in all_valid_tags:
            return tag_full
        # Try to find closest valid dimension name
        for dim in dimensions:
            if tag_name in dim or dim in tag_name:
                prefix = "</" if tag_full.startswith("</") else "<"
                return f"{prefix}{dim}>"
        return tag_full

    xml_content = tag_pattern.sub(_fix_tag_name, xml_content)

    # Step 3: Ensure <response> wrapper exists
    if '<response>' not in xml_content:
        # Check if content starts with a dimension tag
        first_dim_match = None
        for dim in dimensions:
            idx = xml_content.find(f'<{dim}>')
            if idx != -1:
                if first_dim_match is None or idx < first_dim_match:
                    first_dim_match = idx
        if first_dim_match is not None:
            xml_content = xml_content[:first_dim_match] + '<response>' + xml_content[first_dim_match:]

    if '</response>' not in xml_content:
        # Find the last closing dimension tag
        last_close_pos = -1
        for dim in dimensions:
            idx = xml_content.rfind(f'</{dim}>')
            if idx != -1:
                end_pos = idx + len(f'</{dim}>')
                if end_pos > last_close_pos:
                    last_close_pos = end_pos
        if last_close_pos > 0:
            xml_content = xml_content[:last_close_pos] + '</response>' + xml_content[last_close_pos:]

    # Step 4: The critical fix — escape XML-unsafe chars inside <reasoning> blocks.
    # LLM often writes things like "score < 3" or "A & B" inside reasoning text.
    def _escape_reasoning_content(text: str) -> str:
        """Escape XML-unsafe characters in reasoning text content."""
        # Don't escape if it looks like it's already escaped
        if '&lt;' in text or '&gt;' in text or '&amp;' in text:
            return text
        # Escape & first (so we don't double-escape), then < and >
        text = text.replace('&', '&amp;')
        text = text.replace('<', '&lt;')
        text = text.replace('>', '&gt;')
        return text

    for dim in dimensions:
        # Extract reasoning content between <reasoning> and </reasoning> within
        # each dimension block, then escape it
        pattern = re.compile(
            rf'(<{dim}>\s*<reasoning>)(.*?)(</reasoning>)',
            re.DOTALL
        )
        match = pattern.search(xml_content)
        if match:
            before = match.group(1)
            content = match.group(2)
            after = match.group(3)

            # Check if content contains bare < or > that aren't part of valid
            # sub-tags (reasoning should be pure text, no nested XML)
            # Detect if there are "bare" angle brackets that aren't our known tags
            known_inner_tags = re.findall(r'</?(?:reasoning|rating)>', content)
            if not known_inner_tags:
                # No valid sub-tags inside, any < > are text that needs escaping
                has_bare_angles = bool(re.search(r'[<>]', content))
                if has_bare_angles:
                    escaped_content = _escape_reasoning_content(content)
                    xml_content = (
                        xml_content[:match.start()]
                        + before + escaped_content + after
                        + xml_content[match.end():]
                    )

    # Step 5: Handle missing </reasoning> or </rating> using regex reconstruction
    # For each dimension, ensure the structure is:
    #   <dim><reasoning>...</reasoning><rating>...</rating></dim>
    for dim in dimensions:
        dim_open = f'<{dim}>'
        dim_close = f'</{dim}>'

        if dim_open not in xml_content:
            continue

        start_idx = xml_content.find(dim_open)
        end_idx = xml_content.find(dim_close, start_idx)

        if end_idx == -1:
            # Missing closing tag for this dimension — find the next dimension's
            # opening tag or end of <response> and insert before it
            next_boundary = len(xml_content)
            for other_dim in dimensions:
                if other_dim == dim:
                    continue
                other_idx = xml_content.find(f'<{other_dim}>', start_idx + len(dim_open))
                if other_idx != -1 and other_idx < next_boundary:
                    next_boundary = other_idx
            resp_close_idx = xml_content.find('</response>', start_idx)
            if resp_close_idx != -1 and resp_close_idx < next_boundary:
                next_boundary = resp_close_idx
            xml_content = xml_content[:next_boundary] + dim_close + xml_content[next_boundary:]
            end_idx = next_boundary

        # Now check internal structure
        block = xml_content[start_idx:end_idx + len(dim_close)]

        # Check for <reasoning> ... </reasoning>
        if '<reasoning>' in block and '</reasoning>' not in block:
            # Find where <rating> starts or dimension closes
            rating_pos = block.find('<rating>')
            if rating_pos != -1:
                insert_pos = start_idx + rating_pos
                xml_content = xml_content[:insert_pos] + '</reasoning>' + xml_content[insert_pos:]
            else:
                # Insert before dim_close
                close_pos = xml_content.find(dim_close, start_idx)
                xml_content = xml_content[:close_pos] + '</reasoning>' + xml_content[close_pos:]

        # Re-find block boundaries after potential insertion
        start_idx = xml_content.find(dim_open)
        end_idx = xml_content.find(dim_close, start_idx)
        block = xml_content[start_idx:end_idx + len(dim_close)]

        # Check for <rating> ... </rating>
        if '</reasoning>' in block and '<rating>' not in block:
            # Insert <rating></rating> between </reasoning> and </dim>
            reasoning_close_pos = xml_content.find('</reasoning>', start_idx)
            insert_pos = reasoning_close_pos + len('</reasoning>')
            xml_content = (
                xml_content[:insert_pos]
                + '<rating></rating>'
                + xml_content[insert_pos:]
            )
        elif '<rating>' in block and '</rating>' not in block:
            # Insert </rating> before </dim>
            close_pos = xml_content.find(dim_close, start_idx)
            xml_content = xml_content[:close_pos] + '</rating>' + xml_content[close_pos:]

    # Step 6: Remove HTML comments that LLM might include
    xml_content = re.sub(r'<!--.*?-->', '', xml_content, flags=re.DOTALL)

    return xml_content


# ==================== Prompt Builder ====================

class PromptBuilder:
    """Build evaluation prompts."""

    @staticmethod
    def format_conversation_history(messages: List[Dict]) -> str:
        """Format conversation history."""
        history = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls", [])
            
            if role == "system":
                continue
            elif role == "user":
                history.append(f"{role}: {content}")
            elif role == "assistant":
                if tool_calls:
                    for tool_call in tool_calls:
                        tool_name = tool_call.get("name", "unknown")
                        arguments = tool_call.get("arguments", {})
                        history.append(f"assistant tool_calls: {tool_name} ({json.dumps(arguments, ensure_ascii=False, indent=2)})")
                if content:
                    history.append(f"assistant: {content}")
            elif role == "tool":
                tool_name = msg.get("name", "unknown")
                history.append(f"tool_response: {tool_name} {content}")
        
        return "\n".join(history)
    
    @staticmethod
    def get_prompt_template(mode: str, templates: Optional[Dict[str, str]] = None) -> str:
        """Get prompt template based on mode.
        
        This method is designed to be overridden or used with templates parameter.
        When templates is None, it returns an empty string (subclass should override).
        
        Args:
            mode: Evaluation mode ('single-turn' or 'multi-turn')
            templates: Optional dictionary mapping mode to template string
            
        Returns:
            Prompt template string
        """
        if templates is None:
            # Return empty string when no templates provided
            # This allows subclasses to override this method
            raise NotImplementedError(
                "get_prompt_template requires templates parameter or subclass override"
            )
        if mode not in templates:
            raise ValueError(f"Unsupported mode: {mode}")
        return templates[mode]
    
    @staticmethod
    def build_evaluation_prompt(
        trajectory: Dict,
        prompt_template: str,
        tools_schemas: Optional[str] = None
    ) -> str:
        """Build evaluation prompt.
        
        Args:
            trajectory: Trajectory data containing messages, query, context
            prompt_template: Template string with placeholders
            tools_schemas: Optional tool schemas string (for compatibility)
            
        Returns:
            Formatted evaluation prompt
        """
        messages = trajectory.get("messages", [])
        query = trajectory.get("query", "")
        context = trajectory.get("context", "")
        
        conversation_history = PromptBuilder.format_conversation_history(messages)
        
        # Build format kwargs
        format_kwargs = {
            "CONTEXT_INFO": context,
            "QUESTION_CONTENT": query,
            "CONVERSATION_HISTORY": conversation_history
        }
        
        # Add tools_schemas if provided
        if tools_schemas is not None:
            format_kwargs["INTENDED_TOOL"] = tools_schemas
        
        prompt = prompt_template.format(**format_kwargs)
        return prompt


# ==================== Result Parser ====================

class ResultParser:
    """Parse evaluation results."""
    
    RATING_MAP = {
        "极差": 1,
        "较差": 2,
        "一般": 3,
        "较好": 4,
        "优秀": 5
    }
    
    @staticmethod
    def parse_xml_response(xml_content: str, dimensions: List[str]) -> Optional[Dict]:
        """Parse XML format evaluation results with dynamic dimensions."""
        try:
            xml_content = xml_content.strip()
            if xml_content.startswith("```xml"):
                xml_content = xml_content[6:]
            if xml_content.startswith("```"):
                xml_content = xml_content[3:]
            if xml_content.endswith("```"):
                xml_content = xml_content[:-3]
            xml_content = xml_content.strip()

            # Always run fix_xml_tags first — it handles common LLM issues
            # (missing <response>, bare < > in reasoning, missing close tags)
            # and is safe on already-correct XML.
            xml_content = fix_xml_tags(xml_content, dimensions)

            # Extract content within <response> tag (use greedy match to handle
            # cases where reasoning contains angle brackets that survived fixing)
            response_match = re.search(r'<response>(.*)</response>', xml_content, re.DOTALL)
            if response_match:
                xml_content = f"<response>{response_match.group(1)}</response>"

            # Try parsing
            try:
                root = ET.fromstring(xml_content)
            except ET.ParseError:
                # Last resort: try regex-based extraction without XML parsing
                return ResultParser._regex_fallback_parse(xml_content, dimensions)

            result = {}
            for dimension in dimensions:
                result[dimension] = {
                    "reasoning": "",
                    "rating": ""
                }

            for dimension in dimensions:
                element = root.find(dimension)
                if element is not None:
                    reasoning_elem = element.find("reasoning")
                    rating_elem = element.find("rating")

                    if reasoning_elem is not None:
                        result[dimension]["reasoning"] = reasoning_elem.text or ""
                    if rating_elem is not None:
                        result[dimension]["rating"] = (rating_elem.text or "").strip()

            return result

        except ET.ParseError as e:
            print(f"[Error] XML parsing failed: {type(e).__name__}: {e} Retrying...")
            print(f"[Error] Unfixable XML content):\n{xml_content}")
            return None
        except Exception as e:
            print(f"[Error] XML parsing error: {type(e).__name__}: {e} Retrying...")
            print(f"[Error] Unfixable XML content:\n{xml_content}")
            return None

    @staticmethod
    def _regex_fallback_parse(xml_content: str, dimensions: List[str]) -> Optional[Dict]:
        """
        Regex-based fallback parser when ET.fromstring fails even after fix_xml_tags.

        Directly extracts <reasoning> and <rating> content from each dimension
        block using regex, bypassing XML parsing entirely. This handles cases
        where reasoning content contains unfixable XML-breaking characters.
        """
        result = {}
        any_found = False

        for dimension in dimensions:
            result[dimension] = {"reasoning": "", "rating": ""}

            # Try to extract the dimension block
            dim_pattern = re.compile(
                rf'<{dimension}>(.*?)</{dimension}>',
                re.DOTALL
            )
            dim_match = dim_pattern.search(xml_content)
            if not dim_match:
                # Try greedy match as fallback
                dim_pattern_greedy = re.compile(
                    rf'<{dimension}>(.*)</{dimension}>',
                    re.DOTALL
                )
                dim_match = dim_pattern_greedy.search(xml_content)

            if not dim_match:
                continue

            block = dim_match.group(1)

            # Extract reasoning
            reasoning_match = re.search(
                r'<reasoning>(.*?)(?:</reasoning>|<rating>)',
                block,
                re.DOTALL
            )
            if not reasoning_match:
                # Try greedy
                reasoning_match = re.search(
                    r'<reasoning>(.*)',
                    block,
                    re.DOTALL
                )
            if reasoning_match:
                result[dimension]["reasoning"] = reasoning_match.group(1).strip()

            # Extract rating
            rating_match = re.search(
                r'<rating>(.*?)(?:</rating>|$)',
                block,
                re.DOTALL
            )
            if rating_match:
                rating_text = rating_match.group(1).strip()
                # Clean up: only keep the rating keyword
                for valid_rating in ["极差", "较差", "一般", "较好", "优秀"]:
                    if valid_rating in rating_text:
                        rating_text = valid_rating
                        break
                result[dimension]["rating"] = rating_text
                any_found = True

        if any_found:
            return result
        print(
            f"[Error] Regex fallback also failed to extract any ratings. "
            f"Full XML content:\n{xml_content}"
        )
        return None

    @staticmethod
    def rating_to_score(rating: str) -> Optional[int]:
        """Convert text rating to numeric score.
        
        Handles formats like '评级：极差', '极差', '评级:较好' etc.
        """
        if not rating:
            return None
        # Strip prefix like "评级：" or "评级:"
        cleaned = rating.strip()
        for prefix in ["评级：", "评级:"]:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
                break
        # Try exact match first
        score = ResultParser.RATING_MAP.get(cleaned)
        if score is not None:
            return score
        # Fallback: check if any valid rating keyword is contained
        for valid_rating, score_value in ResultParser.RATING_MAP.items():
            if valid_rating in cleaned:
                return score_value
        return None
    
    @staticmethod
    def extract_scores(parsed_result: Optional[Dict], dimensions: List[str]) -> Dict:
        """Extract numeric scores from parsed results."""
        if not parsed_result:
            scores = {f"{dim}_score": None for dim in dimensions}
            scores["average_score"] = None
            return scores
        
        scores = {}
        score_values = []
        
        for dimension in dimensions:
            rating = parsed_result.get(dimension, {}).get("rating", "")
            score = ResultParser.rating_to_score(rating)
            scores[f"{dimension}_score"] = score
            if score is not None:
                score_values.append(score)
        
        if score_values:
            scores["average_score"] = sum(score_values) / len(score_values)
        else:
            scores["average_score"] = None
        
        return scores
    
    @staticmethod
    def parse_meta_judge_response(xml_content: str) -> Optional[Dict]:
        """Parse meta-judge XML response."""
        try:
            xml_content = xml_content.strip()
            if xml_content.startswith("```xml"):
                xml_content = xml_content[6:]
            if xml_content.endswith("```"):
                xml_content = xml_content[:-3]
            xml_content = xml_content.strip()
            
            # Extract content within <meta_evaluation> tag
            response_match = re.search(r'<meta_evaluation>(.*?)</meta_evaluation>', xml_content, re.DOTALL)
            if response_match:
                xml_content = f"<meta_evaluation>{response_match.group(1)}</meta_evaluation>"
            
            root = ET.fromstring(xml_content)
            
            result = {
                "reasoning": "",
                "rating": ""
            }
            
            reasoning_elem = root.find("reasoning")
            rating_elem = root.find("rating")
            
            if reasoning_elem is not None:
                result["reasoning"] = reasoning_elem.text or ""
            if rating_elem is not None:
                result["rating"] = (rating_elem.text or "").strip()
            
            return result
            
        except ET.ParseError as e:
            print(f"[Error] Meta-judge XML parsing failed: {type(e).__name__}: {e} Rertying...")
            return None
        except Exception as e:
            print(f"[Error] Meta-judge parsing error: {type(e).__name__}: {e} Retrying..." )
            import traceback
            traceback.print_exc()
            return None


# ==================== Data Loader ====================

class DataLoader:
    """Load and process data."""
    
    @staticmethod
    def load_trajectories(file_path: str) -> List[Dict]:
        """Load trajectory file."""
        trajectories = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                trajectories = data
            elif isinstance(data, dict):
                if "results" in data:
                    trajectories = data["results"]
                else:
                    trajectories = [data]
            else:
                print(f"[Error] Unsupported JSON format")
                return []
                
        except json.JSONDecodeError as e:
            print(f"[Error] JSON file parsing failed: {type(e).__name__}: {e}")
            return []
        except Exception as e:
            print(f"[Error] File loading failed: {type(e).__name__}: {e}")
            return []
        
        print(f"Successfully loaded {len(trajectories)} trajectories")
        return trajectories
    
    @staticmethod
    def save_results(results: List[Dict], output_file: str):
        """Save evaluation results."""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Results saved to: {output_file}")


# ==================== Statistics Calculator ====================

class StatisticsCalculator:
    """Calculate statistics."""
    
    @staticmethod
    def calculate_statistics(results: List[Dict], dimensions: List[str]) -> Dict:
        """Calculate overall statistics."""
        total = len(results)
        valid_results = [r for r in results if r["evaluation"]["scores"] and r["evaluation"]["scores"].get("average_score") is not None]
        
        if not valid_results:
            return {
                "total_trajectories": total,
                "valid_evaluations": 0,
                "mode": results[0]["mode"] if results else "unknown",
                "dimensions": dimensions,
                "average_scores": {},
                "average_scores_100": {},
                "penalized_average_scores_100": {},
                "final_average_scores_100": {},
                "rating_distribution": {},
                "tool_error_statistics": {},
                "cache_statistics": {},
                "meta_judge_statistics": {}
            }
        
        # Collect dimension scores
        dimension_scores = {dim: [] for dim in dimensions}
        normalized_dimension_scores = {dim: [] for dim in dimensions}
        penalized_dimension_scores = {dim: [] for dim in dimensions}
        final_dimension_scores = {dim: [] for dim in dimensions}
        average_scores = []
        normalized_average_scores = []
        penalized_average_scores = []
        final_average_scores = []
        tool_error_rates = []
        meta_judge_scores = []
        
        # Collect cache hit rate data
        cache_hits_list = []
        cache_misses_list = []
        
        for result in valid_results:
            scores = result["evaluation"]["scores"]
            normalized_scores = result["evaluation"].get("normalized_scores")
            penalized_scores = result["evaluation"].get("penalized_scores")
            final_scores = result["evaluation"].get("final_scores")
            tool_error_rate = result["evaluation"].get("tool_error_rate", 0)
            meta_judge_result = result["evaluation"].get("meta_judge_result")
            
            # Raw scores (1-5 scale)
            for dimension in dimensions:
                score_key = f"{dimension}_score"
                if scores.get(score_key) is not None:
                    dimension_scores[dimension].append(scores[score_key])
            
            if scores.get("average_score") is not None:
                average_scores.append(scores["average_score"])
            
            # Normalized scores (0-100 scale, after normalization only)
            if normalized_scores:
                for dimension in dimensions:
                    score_key = f"{dimension}_score"
                    if normalized_scores.get(score_key) is not None:
                        normalized_dimension_scores[dimension].append(normalized_scores[score_key])
                
                if normalized_scores.get("average_score") is not None:
                    normalized_average_scores.append(normalized_scores["average_score"])
            
            # Penalized scores (0-100 scale, after normalization and tool penalty only)
            if penalized_scores:
                for dimension in dimensions:
                    score_key = f"{dimension}_score"
                    if penalized_scores.get(score_key) is not None:
                        penalized_dimension_scores[dimension].append(penalized_scores[score_key])
                
                if penalized_scores.get("average_score") is not None:
                    penalized_average_scores.append(penalized_scores["average_score"])
            
            # Final scores (0-100 scale, after normalization and all penalties including meta-judge)
            if final_scores:
                for dimension in dimensions:
                    score_key = f"{dimension}_score"
                    if final_scores.get(score_key) is not None:
                        final_dimension_scores[dimension].append(final_scores[score_key])
                
                if final_scores.get("average_score") is not None:
                    final_average_scores.append(final_scores["average_score"])
            
            tool_error_rates.append(tool_error_rate)
            
            # Collect cache hit rate data
            trajectory = result.get("original_trajectory", {})
            cache_hits = trajectory.get("cache_hits", 0)
            cache_misses = trajectory.get("cache_misses", 0)
            cache_hits_list.append(cache_hits)
            cache_misses_list.append(cache_misses)
            
            # Collect meta-judge scores
            if meta_judge_result and meta_judge_result.get("meta_score") is not None:
                meta_judge_scores.append(meta_judge_result["meta_score"])
        
        # Calculate raw average scores (1-5 scale)
        avg_scores = {}
        for dimension in dimensions:
            if dimension_scores[dimension]:
                avg = sum(dimension_scores[dimension]) / len(dimension_scores[dimension])
                avg_scores[dimension] = avg
            else:
                avg_scores[dimension] = 0
        
        avg_overall = sum(average_scores) / len(average_scores) if average_scores else 0
        avg_scores["overall_average"] = avg_overall
        
        # Calculate normalized average scores (0-100 scale)
        avg_scores_100 = {}
        for dimension in dimensions:
            if normalized_dimension_scores[dimension]:
                avg_scores_100[dimension] = sum(normalized_dimension_scores[dimension]) / len(normalized_dimension_scores[dimension])
            else:
                avg_scores_100[dimension] = 0
        
        normalized_avg_overall = sum(normalized_average_scores) / len(normalized_average_scores) if normalized_average_scores else 0
        avg_scores_100["overall_average"] = normalized_avg_overall
        
        # Calculate penalized average scores (0-100 scale, after normalization and tool penalty only)
        penalized_avg_scores_100 = {}
        for dimension in dimensions:
            if penalized_dimension_scores[dimension]:
                penalized_avg_scores_100[dimension] = sum(penalized_dimension_scores[dimension]) / len(penalized_dimension_scores[dimension])
            else:
                penalized_avg_scores_100[dimension] = 0
        
        penalized_avg_overall = sum(penalized_average_scores) / len(penalized_average_scores) if penalized_average_scores else 0
        penalized_avg_scores_100["overall_average"] = penalized_avg_overall
        
        # Calculate final average scores (0-100 scale, after normalization and all penalties including meta-judge)
        final_avg_scores_100 = {}
        for dimension in dimensions:
            if final_dimension_scores[dimension]:
                final_avg_scores_100[dimension] = sum(final_dimension_scores[dimension]) / len(final_dimension_scores[dimension])
            else:
                final_avg_scores_100[dimension] = 0
        
        final_avg_overall = sum(final_average_scores) / len(final_average_scores) if final_average_scores else 0
        final_dimension_100_scores = [final_avg_scores_100[dim] for dim in dimensions if dim in final_avg_scores_100]
        final_avg_scores_100["overall_average"] = sum(final_dimension_100_scores) / len(final_dimension_100_scores) if final_dimension_100_scores else 0
        
        # Tool error rate statistics
        avg_tool_error_rate = sum(tool_error_rates) / len(tool_error_rates) if tool_error_rates else 0
        
        # Cache hit rate statistics
        total_cache_hits = sum(cache_hits_list)
        total_cache_misses = sum(cache_misses_list)
        total_cache_requests = total_cache_hits + total_cache_misses
        cache_hit_rate = (total_cache_hits / total_cache_requests * 100) if total_cache_requests > 0 else 0
        
        cache_statistics = {
            "total_cache_hits": total_cache_hits,
            "total_cache_misses": total_cache_misses,
            "total_cache_requests": total_cache_requests,
            "cache_hit_rate_percentage": cache_hit_rate,
            "average_cache_hits_per_sample": total_cache_hits / len(valid_results) if valid_results else 0,
            "average_cache_misses_per_sample": total_cache_misses / len(valid_results) if valid_results else 0
        }
        
        # Meta-judge statistics
        meta_judge_statistics = {}
        if meta_judge_scores:
            meta_judge_statistics = {
                "enabled": True,
                "total_meta_evaluations": len(meta_judge_scores),
                "average_meta_score": sum(meta_judge_scores) / len(meta_judge_scores),
                "meta_score_distribution": {i: meta_judge_scores.count(i) for i in range(1, 6)}
            }
        else:
            meta_judge_statistics = {
                "enabled": False
            }
        
        # Rating distribution statistics
        rating_distribution = {}
        for dimension in dimensions:
            rating_distribution[dimension] = {i: dimension_scores[dimension].count(i) for i in range(1, 6)}
        
        statistics = {
            "total_trajectories": total,
            "valid_evaluations": len(valid_results),
            "mode": results[0]["mode"] if results else "unknown",
            "dimensions": dimensions,
            "average_scores": avg_scores,
            "average_scores_100": avg_scores_100,
            "penalized_average_scores_100": penalized_avg_scores_100,
            "final_average_scores_100": final_avg_scores_100,
            "rating_distribution": rating_distribution,
            "tool_error_statistics": {
                "average_tool_error_rate": avg_tool_error_rate,
                "total_samples_with_errors": sum(1 for rate in tool_error_rates if rate > 0),
                "error_rate_distribution": {
                    "0%": sum(1 for rate in tool_error_rates if rate == 0),
                    "0-25%": sum(1 for rate in tool_error_rates if 0 < rate <= 0.25),
                    "25-50%": sum(1 for rate in tool_error_rates if 0.25 < rate <= 0.5),
                    "50-75%": sum(1 for rate in tool_error_rates if 0.5 < rate <= 0.75),
                    "75-100%": sum(1 for rate in tool_error_rates if 0.75 < rate <= 1.0)
                }
            },
            "cache_statistics": cache_statistics,
            "meta_judge_statistics": meta_judge_statistics
        }
        
        return statistics


# ==================== POI Category Resolver ====================

from pathlib import Path
from typing import Tuple

# Valid hotel categories (same set used when synthesizing preferences)
_HOTEL_CATEGORIES = {
    "舒适型", "经济型", "高档型", "民宿", "商务酒店",
    "豪华型", "快捷酒店", "旅馆", "客栈", "度假酒店",
    "电竞酒店", "公寓式酒店", "主题酒店",
}

# Attraction / food blacklist (same as EXCLUDED_CATEGORIES in poi_util.py)
_EXCLUDED_CATEGORIES = {
    "attractions": {"旅游景点", "滑雪", "游客中心"},
    "food": {"食堂", "米粉"},
}

# Default fallback category when the blacklist hits
_FALLBACK_CATEGORIES = {
    "attractions": "公园",
    "food": "中餐",
    "hotels": "商务酒店",
}

# Path to the merged POI data
_ALL_POI_PATH = Path(__file__).parent.parent.parent / "poi2category" / "all_poi_by_city.json"


class POICategoryResolver:
    """
    POI category resolver (singleton).

    Given a POI name and city, return its ``custom_category`` (aligned with
    preference synthesis). Data is loaded on first instantiation and reused.

    Usage:
        resolver = POICategoryResolver.get_instance()
        category = resolver.resolve("故宫", "北京市")
    """

    _instance: Optional["POICategoryResolver"] = None

    def __init__(self, data_path: Path = _ALL_POI_PATH):
        self._data_path = data_path
        self._city_pois: Dict[str, List[Dict]] = {}
        # Index: {city: {poi_name: poi_dict}} (exact match)
        self._city_name_index: Dict[str, Dict[str, Dict]] = {}
        # Global name index: {poi_name: poi_dict} (when no city is given)
        self._global_name_index: Dict[str, Dict] = {}
        self._loaded = False

    @classmethod
    def get_instance(cls, data_path: Optional[Path] = None) -> "POICategoryResolver":
        """Return the singleton, loading data on the first call."""
        if cls._instance is None:
            path = data_path if data_path else _ALL_POI_PATH
            cls._instance = cls(path)
            cls._instance._load()
        return cls._instance

    def _load(self):
        """Load ``all_poi_by_city.json`` and build the index."""
        if self._loaded:
            return
        try:
            with open(self._data_path, "r", encoding="utf-8") as f:
                self._city_pois = json.load(f)
        except json.JSONDecodeError as json_error:
            raise RuntimeError(
                f"❌ POI 数据文件 JSON 格式损坏: {self._data_path}\n"
                f"   错误位置: 第 {json_error.lineno} 行, 第 {json_error.colno} 列\n"
                f"   错误信息: {json_error.msg}\n"
                f"   可能原因: 文件在解压过程中被中断导致截断。\n"
                f"   修复方法: 重新运行 bash scripts/restore_data.sh 恢复该文件。"
            ) from json_error
        except FileNotFoundError:
            raise RuntimeError(
                f"❌ POI 数据文件不存在: {self._data_path}\n"
                f"   请先运行 bash scripts/restore_data.sh 恢复评测数据。"
            )

        for city, pois in self._city_pois.items():
            name_index: Dict[str, Dict] = {}
            for poi in pois:
                name = poi.get("name", "")
                if name:
                    name_index[name] = poi
                    # Global index: keep the first entry when names collide
                    if name not in self._global_name_index:
                        self._global_name_index[name] = poi
            self._city_name_index[city] = name_index

        self._loaded = True

    def _match_poi(self, poi_name: str, city: str) -> Optional[Dict]:
        """
        Match a POI in the given city.

        Priority:
          1. Exact match
          2. Substring match (``poi_name`` is a substring of a stored name);
             if several hit, take the highest score ``len(query) / len(full_name)``
        """
        name_index = self._city_name_index.get(city)
        if not name_index:
            return None

        # 1. Exact match
        if poi_name in name_index:
            return name_index[poi_name]

        # 2. Substring match
        best_match: Optional[Dict] = None
        best_ratio: float = 0.0

        for full_name, poi_dict in name_index.items():
            if poi_name in full_name:
                ratio = len(poi_name) / len(full_name)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_match = poi_dict

        return best_match

    def _apply_category_rules(self, custom_category: str, poi_type: str) -> str:
        """
        Apply blacklist and validity rules to ``custom_category``.

        - Hotel: not in ``HOTEL_CATEGORIES`` → "商务酒店"
        - Attraction: hits ``EXCLUDED_CATEGORIES`` → "公园"
        - Food: hits ``EXCLUDED_CATEGORIES`` → "中餐"
        """
        if poi_type == "hotels":
            if custom_category not in _HOTEL_CATEGORIES:
                return _FALLBACK_CATEGORIES["hotels"]
            return custom_category

        excluded = _EXCLUDED_CATEGORIES.get(poi_type, set())
        if custom_category in excluded:
            return _FALLBACK_CATEGORIES.get(poi_type, custom_category)

        return custom_category

    def resolve(self, poi_name: str, city: str) -> Optional[str]:
        """
        Return the ``custom_category`` for a POI name and city.

        Args:
            poi_name: POI name (may be a substring of the full name).
            city: City name (e.g. "北京市").

        Returns:
            ``custom_category`` string, or ``None`` if unmatched.
        """
        poi = self._match_poi(poi_name, city)
        if poi is None:
            return None

        custom_category = poi.get("custom_category", "")
        poi_type = poi.get("poi_type", "unknown")

        if not custom_category:
            return _FALLBACK_CATEGORIES.get(poi_type)

        return self._apply_category_rules(custom_category, poi_type)

    def resolve_by_name(self, poi_name: str) -> Optional[str]:
        """
        Exact-name resolve that does not require a city.

        Looks up the POI in the global name index and returns its
        ``custom_category``. Useful when only the POI name is known
        (e.g. a ``get_poi_detail`` result).

        Args:
            poi_name: Exact POI name.

        Returns:
            ``custom_category`` string, or ``None`` if unmatched.
        """
        poi = self._global_name_index.get(poi_name)
        if poi is None:
            return None

        custom_category = poi.get("custom_category", "")
        poi_type = poi.get("poi_type", "unknown")

        if not custom_category:
            return _FALLBACK_CATEGORIES.get(poi_type)

        return self._apply_category_rules(custom_category, poi_type)

    def resolve_with_detail(
        self, poi_name: str, city: str
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Resolve with extra detail: ``(category, poi_type, matched_full_name)``.

        Useful for debugging or when the matched POI name is needed.
        """
        poi = self._match_poi(poi_name, city)
        if poi is None:
            return None, None, None

        custom_category = poi.get("custom_category", "")
        poi_type = poi.get("poi_type", "unknown")
        matched_name = poi.get("name", "")

        if not custom_category:
            category = _FALLBACK_CATEGORIES.get(poi_type)
        else:
            category = self._apply_category_rules(custom_category, poi_type)

        return category, poi_type, matched_name


# ==================== POI Opening Hours Resolver ====================

from datetime import datetime as _dt

class POIOpeningHoursResolver:
    """
    POI opening-hours resolver (singleton).

    Given a POI name, city, and date, return that day's opening windows.
    Data comes from the preprocessed ``opening_hours`` field in
    ``all_poi_by_city.json``.

    Usage:
        resolver = POIOpeningHoursResolver.get_instance()
        windows = resolver.get_time_windows("故宫博物院", "北京市", "2025-09-23")
        # [{"open": "08:30", "close": "17:00"}] or None (no data / always open)
    """

    _instance: Optional["POIOpeningHoursResolver"] = None

    def __init__(self, data_path: Path = _ALL_POI_PATH):
        self._data_path = data_path
        self._city_name_index: Dict[str, Dict[str, Dict]] = {}
        self._loaded = False

    @classmethod
    def get_instance(cls, data_path: Optional[Path] = None) -> "POIOpeningHoursResolver":
        """Return the singleton instance."""
        if cls._instance is None:
            path = data_path if data_path else _ALL_POI_PATH
            cls._instance = cls(path)
            cls._instance._load()
        return cls._instance

    def _load(self):
        """Load data and build the index."""
        if self._loaded:
            return
        with open(self._data_path, "r", encoding="utf-8") as f:
            city_pois = json.load(f)

        for city, pois in city_pois.items():
            name_index: Dict[str, Dict] = {}
            for poi in pois:
                name = poi.get("name", "")
                if name:
                    name_index[name] = poi
            self._city_name_index[city] = name_index

        self._loaded = True

    def _match_poi(self, poi_name: str, city: str) -> Optional[Dict]:
        """Match a POI (exact, then substring)."""
        name_index = self._city_name_index.get(city)
        if not name_index:
            return None

        if poi_name in name_index:
            return name_index[poi_name]

        best_match: Optional[Dict] = None
        best_ratio: float = 0.0
        for full_name, poi_dict in name_index.items():
            if poi_name in full_name:
                ratio = len(poi_name) / len(full_name)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_match = poi_dict
        return best_match

    def get_time_windows(
        self, poi_name: str, city: str, date_str: str
    ) -> Optional[List[Dict[str, str]]]:
        """
        Return the POI's opening windows on a given date.

        Args:
            poi_name: POI name.
            city: City name.
            date_str: Date string, ``YYYY-MM-DD``.

        Returns:
            List of windows ``[{"open": "HH:MM", "close": "HH:MM"}]``.
            ``None`` means no data or always open (no check needed).
        """
        poi = self._match_poi(poi_name, city)
        if poi is None:
            return None

        opening_hours = poi.get("opening_hours")
        if opening_hours is None:
            return None

        if opening_hours.get("always_open", False):
            return None

        rules = opening_hours.get("rules", [])
        if not rules:
            return None

        # Parse the date
        try:
            date_obj = _dt.strptime(date_str, "%Y-%m-%d")
        except (ValueError, TypeError):
            return None

        # weekday: 1=Monday ... 7=Sunday
        weekday = date_obj.isoweekday()
        month_day = date_obj.strftime("%m-%d")

        # Find the matching rule
        for rule in rules:
            date_range = rule.get("date_range")
            weekdays = rule.get("weekdays", [1, 2, 3, 4, 5, 6, 7])
            time_windows = rule.get("time_windows", [])

            # Check the date range
            if date_range:
                range_start = date_range.get("start", "01-01")
                range_end = date_range.get("end", "12-31")

                if range_start <= range_end:
                    if not (range_start <= month_day <= range_end):
                        continue
                else:
                    # Range that crosses the year boundary (e.g. 11-01 to 03-31)
                    if not (month_day >= range_start or month_day <= range_end):
                        continue

            # Check the weekday
            if weekday not in weekdays:
                # Weekday is not in the open set → empty list means closed that day
                return []

            if time_windows:
                return time_windows

        # No matching rule; return None (cannot decide)
        return None

    def is_within_opening_hours(
        self,
        poi_name: str,
        city: str,
        date_str: str,
        activity_start: str,
        activity_end: str,
    ) -> Optional[bool]:
        """
        Return whether an activity falls inside the POI's opening windows.

        Args:
            poi_name: POI name.
            city: City name.
            date_str: Date ``YYYY-MM-DD``.
            activity_start: Activity start ``HH:MM``.
            activity_end: Activity end ``HH:MM``.

        Returns:
            True: inside opening hours
            False: outside opening hours
            None: unknown (no data / always open)
        """
        windows = self.get_time_windows(poi_name, city, date_str)

        if windows is None:
            return None

        if not windows:
            # An empty list means closed that day
            return False

        # Check that the visit falls entirely inside an opening window
        for window in windows:
            window_open = window.get("open", "")
            window_close = window.get("close", "")

            if not window_open or not window_close:
                continue

            if activity_start >= window_open and activity_end <= window_close:
                return True

        return False
