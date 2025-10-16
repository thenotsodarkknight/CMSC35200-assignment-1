#!/usr/bin/env python3
"""
Task 1: Multi-model "Game of Telephone" using the Argonne ALCF Sophia inference service.

Each input prompt is paraphrased sequentially by four models. We log intermediate
outputs, latencies, and final paraphrases. Results are persisted as JSON and markdown.
"""
import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

from openai import APIConnectionError, APITimeoutError, InternalServerError, OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent
SOPHIA_TOOLS = PROJECT_ROOT.parent / "Sophia-tools"
if str(SOPHIA_TOOLS) not in sys.path:
    sys.path.append(str(SOPHIA_TOOLS))

from inference_auth_token import get_access_token  # type: ignore  # pylint: disable=wrong-import-position


DEFAULT_MODELS = [
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "openai/gpt-oss-20b",
    "google/gemma-3-27b-it",
    "meta-llama/Meta-Llama-3.1-70B-Instruct",
]

DEFAULT_PROMPTS = [
    "Explain how photosynthesis powers life on Earth and why it matters to humans.",
    "Summarize the plot of Shakespeare's Hamlet in three sentences for a teenager.",
    "Describe a futuristic transportation system that solves urban congestion.",
    "Outline a day in the life of a Martian botanist growing crops in a habitat dome.",
    "Teach me the basics of quantum entanglement using a cooking metaphor.",
    "Imagine an AI assistant helping emergency responders during a hurricane.",
    "Propose a community project that reduces food waste and supports shelters.",
    "Explain CRISPR gene editing to someone who has never heard of DNA.",
    "Describe the evolution of jazz music from its roots to modern interpretations.",
    "Persuade a city council to invest in green rooftop gardens for public buildings.",
]


@dataclass
class StageResult:
    model: str
    latency_sec: float
    output_text: str


@dataclass
class TelephoneRun:
    input_prompt: str
    stages: List[StageResult]

    @property
    def final_output(self) -> str:
        return self.stages[-1].output_text if self.stages else ""


def build_client(timeout: int) -> OpenAI:
    access_token = get_access_token()
    return OpenAI(
        api_key=access_token,
        base_url="https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1",
        timeout=timeout,
    )


def paraphrase_message(client: OpenAI, model: str, text: str, timeout: int, retry: int = 2) -> StageResult:
    paraphrase_prompt = (
        "Paraphrase the following message. Keep the core meaning but change tone, word choice, "
        "and sentence structure. Limit the response to 200 words. Message:\n\n"
        f"{text}"
    )
    for attempt in range(retry + 1):
        start = time.perf_counter()
        try:
            response = client.with_options(timeout=timeout).chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": paraphrase_prompt}],
            )
        except (APITimeoutError, APIConnectionError, InternalServerError) as exc:
            if attempt >= retry:
                raise RuntimeError(f"Model {model} failed after {retry + 1} attempts: {exc}") from exc
            print(f"    ! Model {model} attempt {attempt + 1} failed: {exc}. Retrying...", flush=True)
            time.sleep(1.5 * (attempt + 1))
            continue

        latency = time.perf_counter() - start
        message = response.choices[0].message.content.strip()
        return StageResult(model=model, latency_sec=latency, output_text=message)

    raise RuntimeError(f"Model {model} did not return after retries.")


def run_telephone(prompts: List[str], models: List[str], timeout: int) -> List[TelephoneRun]:
    client = build_client(timeout=timeout)
    runs: List[TelephoneRun] = []
    total_prompts = len(prompts)
    for prompt_idx, prompt in enumerate(prompts, start=1):
        print(f"[Prompt {prompt_idx}/{total_prompts}] Starting run", flush=True)
        stages: List[StageResult] = []
        current_text = prompt
        for stage_idx, model in enumerate(models, start=1):
            print(f"    -> Stage {stage_idx}: requesting {model}", flush=True)
            stage = paraphrase_message(client, model, current_text, timeout=timeout)
            stages.append(stage)
            current_text = stage.output_text
            print(
                f"    <- Stage {stage_idx}: {model} returned {len(stage.output_text)} chars "
                f"in {stage.latency_sec:.2f}s",
                flush=True,
            )
        runs.append(TelephoneRun(input_prompt=prompt, stages=stages))
        print(f"[Prompt {prompt_idx}/{total_prompts}] Completed run\n", flush=True)
    return runs


def save_results(runs: List[TelephoneRun], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "telephone_runs.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(
            [
                {
                    "input_prompt": run.input_prompt,
                    "stages": [asdict(stage) for stage in run.stages],
                    "final_output": run.final_output,
                }
                for run in runs
            ],
            fh,
            indent=2,
        )

    markdown_path = output_dir / "telephone_runs.md"
    with markdown_path.open("w", encoding="utf-8") as fh:
        fh.write("# Game of Telephone Results\n\n")
        for idx, run in enumerate(runs, start=1):
            fh.write(f"## Prompt {idx}\n")
            fh.write(f"**Input:** {run.input_prompt}\n\n")
            for stage_idx, stage in enumerate(run.stages, start=1):
                fh.write(f"- **Stage {stage_idx} ({stage.model} | {stage.latency_sec:.2f}s):** {stage.output_text}\n")
            fh.write(f"\n**Final Output:** {run.final_output}\n\n")

    print(f"Saved JSON results to {json_path}")
    print(f"Saved Markdown summary to {markdown_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a multi-model telephone paraphrasing experiment.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Sequence of models to use for each telephone run.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=45,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "telephone",
        help="Directory where results will be written.",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        help="Optional path to a text file containing one prompt per line. Overrides the default prompts.",
    )
    return parser.parse_args()


def load_prompts(args: argparse.Namespace) -> List[str]:
    if args.prompt_file:
        data = args.prompt_file.read_text(encoding="utf-8").strip().splitlines()
        prompts = [line.strip() for line in data if line.strip()]
        if not prompts:
            raise ValueError(f"No prompts found in {args.prompt_file}")
        return prompts
    return DEFAULT_PROMPTS


def main() -> None:
    args = parse_args()
    prompts = load_prompts(args)
    print(f"Running telephone experiment with {len(prompts)} prompts across models: {args.models}")
    runs = run_telephone(prompts, args.models, timeout=args.timeout)
    save_results(runs, args.output_dir)


if __name__ == "__main__":
    main()

