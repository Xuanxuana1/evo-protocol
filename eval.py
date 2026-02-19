"""Evaluation Script - Using OpenAI API for Grading

Use GPT or other LLMs as the judge to grade model outputs with binary scores (0/1).

Input File:
    JSONL file with model outputs, each line contains:
    {"idx": 0, "messages": [...], "model_output": "...", "ref_answer": "...", "rubrics": [...]}

Output File:
    outputs/{model_name}_graded.jsonl

Usage:
    # Using default OpenAI API
    python eval.py --input outputs/model_output.jsonl --output outputs/model_graded.jsonl
    
    # Using other compatible APIs
    python eval.py --input outputs/model_output.jsonl --base-url https://api.deepseek.com/v1 --api-key your_key
    
    # Concurrent evaluation
    python eval.py --input outputs/model_output.jsonl --workers 5
"""

import json
import os
import argparse
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI
from core.env_utils import env_float, env_int, first_env, load_env_file


def get_timestamp():
    """Get current timestamp string."""
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def log(message):
    """Print log message with timestamp."""
    print(f"[{get_timestamp()}] {message}")


def load_jsonl(file_path):
    """Load JSONL file."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def append_jsonl(item, file_path):
    """Append a single record to JSONL file."""
    os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else ".", exist_ok=True)
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def build_rubrics_text(rubrics):
    """Build rubrics checklist from rubrics list."""
    if not rubrics:
        return "No specific rubrics provided."
    
    lines = []
    for i, rubric in enumerate(rubrics, 1):
        if isinstance(rubric, dict):
            criteria = rubric.get("rubric_criteria", "").strip()
        else:
            criteria = str(rubric).strip()
        if criteria:
            lines.append(f"{i}. {criteria}")
    
    return "\n".join(lines) if lines else "No specific rubrics provided."


def call_judge_api(
    client,
    model,
    rubrics_text,
    model_output,
    max_retries=3,
    retry_delay=3,
    request_timeout=90.0,
    max_tokens=None,
):
    """
    Call judge model API for grading (only handles API call, returns raw text).
    
    Args:
        client: OpenAI client instance
        model: Judge model name
        rubrics_text: Formatted rubrics text
        model_output: Model's response to be graded
        max_retries: Maximum number of retries for API call
        retry_delay: Delay between retries (seconds)
    
    Returns:
        result_text: Raw response text from API, or None if failed
    """
    grading_prompt = (
        "Starting now, you are a rigorous instruction-following grading teacher. Your task is to accurately grade and score student answers based on the 【Rubrics】.\n\n"
        "Grading Criteria\n"
        "This is a strict, all-or-nothing grading system. The final score is binary.\n"
        "To receive a score of 1, the student's answer must perfectly satisfy every single requirement listed in the 【Rubrics】.\n"
        "If even one requirement is not fully met, the final score will be 0.\n"
        "Grading Process\n"
        "Please strictly follow the steps below for analysis—no steps may be skipped:\n"
        "Step 1: Analyze the Standard Answer\n"
        "List all explicit requirements in the 【Rubrics】 item by item (including format, content, quantity, order, etc.).\n"
        "Identify implicit requirements in the 【Rubrics】 (e.g., language style, logical structure).\n"
        "Define specific evaluation criteria for each requirement (e.g., \"must include X,\" \"must not exceed Y\").\n"
        "Step 2: Check Each Requirement Against the Student's Answer\n"
        "For every requirement in the 【Rubrics】, verify one by one whether the student's answer fully satisfies it.\n"
        "Step 3: Self-Reflection\n"
        "Before giving the final score, you must conduct the following checks:\n"
        "  Completeness Check: Whether all requirements in the standard answer have been reviewed with no omissions.\n"
        "  Strictness Check: Whether the evaluation strictly adheres to the \"fully satisfied\" standard without relaxing requirements due to subjective judgment.\n"
        "  Consistency Check: Whether the grading rationale aligns logically with the final score.\n"
        "  Objectivity Check: Whether judgments are based on objective facts rather than subjective speculation.\n"
        "Output Format Requirements\n"
        "【Grading Rationale】: xxx\n"
        "【List of Requirement Satisfaction Status】: [x₁, x₂, …, xᵢ, …, xₙ] (where n is the total number of requirements in the 【Rubrics】, and xᵢ indicates whether the student's answer meets the i-th requirement, with values \"yes\"/\"no\")\n"
        "【Overall Score】: x points (x is an integer, either 0 or 1.)\n\n"
        "Content to Be Graded\n"
        f"【Rubrics】:\n{rubrics_text}\n"
        f"【Student Response】:\n{model_output}\n"
        "\nPlease strictly output ONLY the following JSON format (do not output any other content):\n"
        "{\n"
        '  "Grading Rationale": "Your detailed grading rationale",\n'
        '  "List of Requirement Satisfaction Status": ["yes", "no", ...],\n'
        '  "Overall Score": 0 or 1\n'
        "}\n"
    )
    
    messages = [{"role": "user", "content": grading_prompt}]
    resolved_max_tokens = env_int(
        ["JUDGE_MAX_TOKENS", "EVAL_MAX_TOKENS", "OPENAI_MAX_TOKENS", "MAX_TOKENS"],
        default=65536,
    )
    if max_tokens is not None:
        try:
            parsed_max_tokens = int(max_tokens)
            if parsed_max_tokens > 0:
                resolved_max_tokens = parsed_max_tokens
        except (TypeError, ValueError):
            pass
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=resolved_max_tokens,
                timeout=request_timeout,
            )
            result_text = response.choices[0].message.content.strip()
            
            # Remove code block wrapper if present
            if result_text.startswith("```json"):
                result_text = result_text[7:]
            if result_text.startswith("```"):
                result_text = result_text[3:]
            if result_text.endswith("```"):
                result_text = result_text[:-3]
            result_text = result_text.strip()
            
            return result_text
                
        except Exception as e:
            error_msg = str(e)
            if attempt < max_retries - 1:
                log(f"   ⚠️ API call failed (attempt {attempt + 1}/{max_retries}): {error_msg[:100]}")
                time.sleep(retry_delay)
            else:
                log(f"   ❌ API call failed after {max_retries} attempts: {error_msg[:100]}")
                return None
    
    return None


def process_single_item(args):
    """Process a single item for grading."""
    item, client, judge_model, max_retries, request_timeout, max_tokens = args
    idx = item.get("idx", -1)
    
    model_output = item.get("model_output", "")
    rubrics = item.get("rubrics", [])
    
    # Skip if no model output
    if not model_output or not model_output.strip():
        result = {
            **item,
            "grading_rationale": "No model output (counted as score 0)",
            "requirement_status": [],
            "score": 0
        }
        return idx, result, None # None is no error
    
    # Build rubrics text
    rubrics_text = build_rubrics_text(rubrics)
    
    # JSON parsing retry logic (re-call API if JSON parsing fails)
    for parse_attempt in range(max_retries):
        # Call judge API
        grading_result = call_judge_api(
            client,
            judge_model,
            rubrics_text,
            model_output,
            max_retries=max_retries,
            request_timeout=request_timeout,
            max_tokens=max_tokens,
        )
        
        if not grading_result:
            log(f"   ❌ [idx={idx}] API call failed (attempt {parse_attempt + 1}/{max_retries})")
            if parse_attempt < max_retries - 1:
                log(f"      Waiting 2s before retry...")
                time.sleep(2)
                continue
            else:
                # All retries failed
                result = {
                    **item,
                    "grading_rationale": "API call failed (counted as score 0)",
                    "requirement_status": [],
                    "score": 0
                }
                return idx, result, "API call failed" # error
        
        # Try to parse JSON
        try:
            result_json = json.loads(grading_result)
            
            # Validate required field
            if "Overall Score" not in result_json:
                raise ValueError("Missing 'Overall Score' field")
            
            # Parse success
            result = {
                **item,
                "grading_rationale": result_json.get("Grading Rationale", ""),
                "requirement_status": result_json.get("List of Requirement Satisfaction Status", []),
                "score": result_json.get("Overall Score", "")
            }
            return idx, result, None # None is no error
            
        except (json.JSONDecodeError, ValueError) as e:
            log(f"   ⚠️ [idx={idx}] JSON parse failed (attempt {parse_attempt + 1}/{max_retries}): {e}")
            log(f"      Raw response: {grading_result[:200]}...")
            
            if parse_attempt < max_retries - 1:
                log(f"      Waiting 2s before re-grading...")
                time.sleep(2)
            else:
                log(f"   ❌ [idx={idx}] JSON parse failed after {max_retries} attempts")
                result = {
                    **item,
                    "grading_rationale": f"JSON parse failed ({max_retries} attempts): {grading_result[:500]}",
                    "requirement_status": [],
                    "score": 0
                }
                return idx, result, f"JSON parse failed: {e}" # error
    
    # Should not reach here
    result = {
        **item,
        "grading_rationale": "Unknown error (counted as score 0)",
        "requirement_status": [],
        "score": 0
    }
    return idx, result, "Unknown error"


def main():
    parser = argparse.ArgumentParser(description="Evaluation Script - OpenAI API Judge")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file path")
    parser.add_argument("--output", type=str, default=None, help="Output JSONL file path")
    parser.add_argument("--judge-model", type=str, default=None, help="Judge model name")
    parser.add_argument("--env-file", type=str, default=".env", help="Path to env file (default: .env)")
    parser.add_argument("--base-url", type=str, default=None, help="API Base URL (optional)")
    parser.add_argument("--api-key", type=str, default=None, help="API Key (optional)")
    parser.add_argument("--workers", type=int, default=1, help="Number of concurrent workers")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retries per item")
    parser.add_argument("--api-timeout", type=float, default=None, help="Per-request API timeout in seconds")
    parser.add_argument("--max-tokens", type=int, default=None, help="Max completion tokens per judge call")
    args = parser.parse_args()

    load_env_file(args.env_file)
    args.judge_model = (
        args.judge_model
        or first_env(["EVAL_JUDGE_MODEL", "OPENAI_JUDGE_MODEL", "JUDGE_MODEL", "MODEL"])
        or "gpt-5.1"
    )
    api_timeout = (
        float(args.api_timeout)
        if args.api_timeout and args.api_timeout > 0
        else env_float(
            ["JUDGE_API_TIMEOUT_SECONDS", "OPENAI_API_TIMEOUT_SECONDS", "OPENAI_API_TIMEOUT", "API_TIMEOUT_SECONDS"],
            default=90.0,
        )
    )
    max_tokens = (
        int(args.max_tokens)
        if args.max_tokens and int(args.max_tokens) > 0
        else env_int(
            ["JUDGE_MAX_TOKENS", "EVAL_MAX_TOKENS", "OPENAI_MAX_TOKENS", "MAX_TOKENS"],
            default=65536,
        )
    )
    resolved_base_url = args.base_url or first_env(["OPENAI_BASE_URL", "BASE_URL", "OPENAI_API_BASE"])
    api_key = args.api_key or first_env(["OPENAI_API_KEY", "API_KEY"])
    
    # Set output path
    if args.output is None:
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        args.output = f"outputs/{base_name}_graded.jsonl"
    
    log("=" * 60)
    log("🎯 Evaluation Task")
    log("=" * 60)
    log(f"📥 Input file: {args.input}")
    log(f"📤 Output file: {args.output}")
    log(f"🤖 Judge model: {args.judge_model}")
    log(f"⚡ Workers: {args.workers}")
    log(f"⏱️ API timeout: {api_timeout:.1f}s")
    log(f"🧱 Max tokens: {max_tokens}")
    log("=" * 60)
    
    # Initialize OpenAI client
    if not api_key:
        log("❌ Error: Please set OPENAI_API_KEY/API_KEY or use --api-key argument")
        return
    
    client_kwargs = {"api_key": api_key}
    if resolved_base_url:
        client_kwargs["base_url"] = resolved_base_url
        log(f"🔗 Using custom API: {resolved_base_url}")
    client_kwargs["timeout"] = api_timeout

    client = OpenAI(**client_kwargs)
    
    # Load data
    log("📖 Loading data...")
    data = load_jsonl(args.input)
    log(f"   Total {len(data)} samples")
    
    # Check completed samples (resume from checkpoint)
    completed_indices = set()
    if os.path.exists(args.output):
        existing_data = load_jsonl(args.output)
        completed_indices = {item.get("idx") for item in existing_data if item.get("idx") is not None}
        log(f"📌 Found {len(completed_indices)} completed, resuming remaining")
    
    # Filter pending tasks
    pending_items = [item for item in data if item.get("idx") not in completed_indices]
    
    if not pending_items:
        log("✅ All samples already evaluated")
        # Calculate final statistics
        calculate_statistics(args.output)
        return
    
    log(f"🚀 Starting evaluation ({len(pending_items)} pending)...")
    
    # Prepare tasks
    tasks = [
        (item, client, args.judge_model, args.max_retries, api_timeout, max_tokens)
        for item in pending_items
    ]
    
    # Statistics
    success_count = 0
    fail_count = 0
    
    if args.workers == 1:
        # Single-threaded
        for task in tqdm(tasks, desc="Evaluating"):
            idx, result, error = process_single_item(task)
            
            if error:
                fail_count += 1
            else:
                append_jsonl(result, args.output)
                success_count += 1
    else:
        # Multi-threaded
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_single_item, task): task[0].get("idx") for task in tasks}
            
            with tqdm(total=len(tasks), desc="Evaluating") as pbar:
                for future in as_completed(futures):
                    try:
                        idx, result, error = future.result()
                        
                        if error:
                            fail_count += 1
                        else:
                            append_jsonl(result, args.output)
                            success_count += 1
                    except Exception as e:
                        log(f"   ❌ Exception: {str(e)}")
                        fail_count += 1
                    pbar.update(1)
    
    # Summary
    log("=" * 60)
    log(f"✅ Evaluation completed!")
    log(f"   Success: {success_count}")
    log(f"   Failed: {fail_count}")
    log(f"   Output: {args.output}")
    
    # Calculate final statistics
    calculate_statistics(args.output)


def calculate_statistics(output_path):
    """Calculate and display final statistics."""
    if not os.path.exists(output_path):
        return
    
    data = load_jsonl(output_path)
    
    total = len(data)
    score_0 = sum(1 for item in data if item.get("score") == 0)
    score_1 = sum(1 for item in data if item.get("score") == 1)
    
    log("\n📊 Final Statistics:")
    log(f"   Total samples: {total}")
    log(f"   Score 0: {score_0}")
    log(f"   Score 1: {score_1}")
    
    if total > 0:
        solving_rate = score_1 / total
        log(f"\n📈 Solving Rate: {solving_rate:.4f} ({score_1}/{total})")
    
    log("=" * 60)


if __name__ == "__main__":
    main()
