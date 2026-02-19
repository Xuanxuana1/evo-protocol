# Repository Guidelines

## Project Structure & Module Organization
- `infer.py`: runs model inference over JSONL tasks and writes predictions.
- `eval.py`: grades model outputs against rubrics and reports solving rate.
- `data/`: benchmark files (`CL-bench.jsonl`, license, dataset README).
- `assets/`: figures used in documentation; `docs/`: static leaderboard site.
- `outputs/` (generated): local inference/evaluation artifacts; keep this out of commits unless explicitly needed.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate`: create and activate an isolated environment.
- `pip install -r requirements.txt`: install runtime dependencies (`openai`, `tqdm`).
- `python infer.py --model gpt-5.1 --input data/CL-bench.jsonl --output outputs/gpt5-1.jsonl --max-samples 20`: quick inference smoke run.
- `python eval.py --input outputs/gpt5-1.jsonl --judge-model gpt-5.1`: grade outputs and compute solving rate.
- `python -m py_compile infer.py eval.py`: fast syntax check before opening a PR.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation and clear, small functions.
- Use `snake_case` for variables/functions and descriptive CLI argument names.
- Preserve existing JSONL field names (`messages`, `rubrics`, `metadata`, `score`) for compatibility.
- Keep logging concise and timestamped, matching current script style.

## Testing Guidelines
- There is no formal test suite yet; run targeted smoke tests with `--max-samples`.
- For parser/retry logic changes, validate both single-worker and multi-worker paths.
- If adding reusable logic, introduce `pytest` tests under `tests/` and name files `test_<module>.py`.

## Commit & Pull Request Guidelines
- This checkout has no accessible `.git` history; use Conventional Commit style (e.g., `feat: add retry backoff for judge parsing`).
- Keep commits focused (one behavior change per commit) and reference affected scripts.
- PRs should include: purpose, exact commands run, sample output path(s), and any metric changes.
- Attach screenshots only for `docs/` UI updates.

## Security & Configuration Tips
- Set credentials via environment variables (for example, `OPENAI_API_KEY`); never hardcode or commit keys.
- Respect dataset license terms in `data/LICENSE.txt` (evaluation/benchmarking use only).
