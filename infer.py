"""Inference Script - Using Standard OpenAI API

Process message-format JSONL data and call OpenAI-compatible APIs for inference.

Input File:
    CL-bench.jsonl - Each line contains {"messages": [...], "rubrics": [...], "metadata": {...}}

Output File:
    outputs/{model_name}.jsonl

Usage:
    # Using default OpenAI API
    python infer.py --model gpt-5.1 --input CL-bench.jsonl --output outputs/gpt5-1.jsonl
    
    # Using other compatible APIs (e.g., DeepSeek, Qwen, etc.)
    python infer.py --model deepseek-chat --base-url https://api.deepseek.com/v1 --api-key your_key
    
    # Concurrent inference
    python infer.py --model gpt-5.1 --workers 5
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


def save_jsonl(data, file_path):
    """Save data to JSONL file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def append_jsonl(item, file_path):
    """Append a single record to JSONL file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def call_openai_api(
    client,
    messages,
    model,
    max_retries=3,
    retry_delay=3,
    request_timeout=90.0,
    max_tokens=65536,
):
    """
    Call OpenAI-compatible API.
    
    Args:
        client: OpenAI client instance
        messages: List of messages
        model: Model name
        max_retries: Maximum number of retries
        retry_delay: Delay between retries (seconds)
    
    Returns:
        response_text: Model response text
        error: Error message (if any)
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                timeout=request_timeout,
            )
            return response.choices[0].message.content, None
        except Exception as e:
            error_msg = str(e)
            if attempt < max_retries - 1:
                log(f"   ⚠️ Call failed (attempt {attempt + 1}): {error_msg[:100]}")
                log(f"   ⏳ Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
            else:
                log(f"   ❌ Final failure: {error_msg[:200]}")
                return None, error_msg
    
    return None, "Unknown error"


def process_single_case(args):
    """Process a single data sample."""
    idx, item, client, model, request_timeout, retry_delay, max_tokens = args
    
    # Get messages
    messages = item.get("messages") 
    
    if not messages:
        return idx, None, "No messages found"
    
    # Call API
    response_text, error = call_openai_api(
        client,
        messages,
        model,
        retry_delay=retry_delay,
        request_timeout=request_timeout,
        max_tokens=max_tokens,
    )
    
    if error:
        return idx, None, error
    
    # Build output
    result = {
        "idx": idx,
        "messages": messages,
        "model_output": response_text,
        "rubrics": item.get("rubrics", [])
    }
    
    return idx, result, None


def main():
    parser = argparse.ArgumentParser(description="Simple Inference Script - OpenAI API")
    parser.add_argument("--model", type=str, default=None, help="Model name")
    parser.add_argument("--input", type=str, default="CL-bench.jsonl", help="Input file path")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    parser.add_argument("--env-file", type=str, default=".env", help="Path to env file (default: .env)")
    parser.add_argument("--base-url", type=str, default=None, help="API Base URL (optional)")
    parser.add_argument("--api-key", type=str, default=None, help="API Key (optional, defaults to env var)")
    parser.add_argument("--workers", type=int, default=1, help="Number of concurrent workers")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to process (for testing)")
    parser.add_argument("--retry-delay", type=int, default=3, help="Retry delay in seconds")
    parser.add_argument("--api-timeout", type=float, default=None, help="Per-request API timeout in seconds")
    parser.add_argument("--max-tokens", type=int, default=None, help="Max completion tokens per inference call")
    args = parser.parse_args()

    load_env_file(args.env_file)

    args.model = args.model or first_env(["INFER_MODEL", "OPENAI_MODEL", "MODEL"]) or "gpt-5.1"
    api_timeout = (
        float(args.api_timeout)
        if args.api_timeout and args.api_timeout > 0
        else env_float(
            ["INFER_API_TIMEOUT_SECONDS", "OPENAI_API_TIMEOUT_SECONDS", "OPENAI_API_TIMEOUT", "API_TIMEOUT_SECONDS"],
            default=90.0,
        )
    )
    max_tokens = (
        int(args.max_tokens)
        if args.max_tokens and int(args.max_tokens) > 0
        else env_int(
            ["INFER_MAX_TOKENS", "OPENAI_MAX_TOKENS", "MAX_TOKENS"],
            default=65536,
        )
    )
    resolved_base_url = args.base_url or first_env(["OPENAI_BASE_URL", "BASE_URL", "OPENAI_API_BASE"])
    api_key = args.api_key or first_env(["OPENAI_API_KEY", "API_KEY"])
    
    # Set output path
    if args.output is None:
        model_name_safe = args.model.replace("/", "_").replace(":", "_")
        args.output = f"outputs/{model_name_safe}.jsonl"
    
    log(f"📂 Input file: {args.input}")
    log(f"📂 Output file: {args.output}")
    log(f"🤖 Model: {args.model}")
    log(f"🔧 Workers: {args.workers}")
    log(f"⏱️ API timeout: {api_timeout:.1f}s")
    log(f"🧱 Max tokens: {max_tokens}")
    
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
    
    if args.max_samples:
        data = data[:args.max_samples]
        log(f"   Limited to {args.max_samples} samples")
    
    # Check completed samples (resume from checkpoint)
    completed_indices = set()
    if os.path.exists(args.output):
        existing_data = load_jsonl(args.output)
        completed_indices = {item.get("idx") for item in existing_data if item.get("idx") is not None}
        log(f"📌 Found {len(completed_indices)} completed, resuming remaining")
    
    # Filter pending tasks
    tasks = [
        (idx, item, client, args.model, api_timeout, args.retry_delay, max_tokens)
        for idx, item in enumerate(data)
        if idx not in completed_indices
    ]
    
    if not tasks:
        log("✅ All samples already processed")
        return
    
    log(f"🚀 Starting inference ({len(tasks)} pending)...")
    
    # Statistics
    success_count = 0
    fail_count = 0
    
    if args.workers == 1:
        # Single-threaded sequential execution
        for task in tqdm(tasks, desc="Inference"):
            idx, result, error = process_single_case(task)
            if result:
                append_jsonl(result, args.output)
                success_count += 1
            else:
                log(f"   ❌ Sample {idx} failed: {error}")
                fail_count += 1
    else:
        # Multi-threaded concurrent execution
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_single_case, task): task[0] for task in tasks}
            
            with tqdm(total=len(tasks), desc="Inference") as pbar:
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        idx, result, error = future.result()
                        if result:
                            append_jsonl(result, args.output)
                            success_count += 1
                        else:
                            log(f"   ❌ Sample {idx} failed: {error}")
                            fail_count += 1
                    except Exception as e:
                        log(f"   ❌ Sample {idx} exception: {str(e)}")
                        fail_count += 1
                    pbar.update(1)
    
    # Summary
    log("=" * 50)
    log(f"✅ Inference completed!")
    log(f"   Success: {success_count}")
    log(f"   Failed: {fail_count}")
    log(f"   Output: {args.output}")


if __name__ == "__main__":
    main()
