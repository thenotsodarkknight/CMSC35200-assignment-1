#!/usr/bin/env python3
"""
Task 3: Gene-disease relationship scan via the Sophia OpenAI-compatible endpoint.

Workflow
--------
1. Download the HGNC complete gene list (if not present).
2. Randomly sample 50 human genes.
3. Query the specified LLM to classify each gene for links to cancer, heart disease,
   diabetes, and dementia, plus suggest interactions among the selected genes.
4. Persist raw model output, a structured JSON report, and timing metadata.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import requests
from openai import OpenAI
from json_repair import repair_json

# Sophia tooling for access tokens.
PROJECT_ROOT = Path(__file__).resolve().parent
SOPHIA_TOOLS = PROJECT_ROOT.parent / "Sophia-tools"
import sys

if str(SOPHIA_TOOLS) not in sys.path:
    sys.path.append(str(SOPHIA_TOOLS))

from inference_auth_token import get_access_token  # type: ignore  # pylint: disable=wrong-import-position

GENESET_URL = (
    "https://www.genenames.org/cgi-bin/download/custom?"
    "col=gd_app_sym&status=Approved&hgnc_dbtag=on&order_by=gd_app_sym_sort&format=text&submit=submit"
)
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "gene_analysis"
DEFAULT_MODEL = "meta-llama/Meta-Llama-3.1-70B-Instruct"
MANUAL_MINUTES_PER_GENE = 4.0  # Conservative manual research estimate


@dataclass
class GeneResult:
    gene: str
    has_cancer_link: bool
    has_heart_disease_link: bool
    has_diabetes_link: bool
    has_dementia_link: bool
    explanation: str
    interacting_genes: List[str]


def ensure_gene_catalog() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    local_path = DATA_DIR / "approved_gene_symbols.txt"
    if local_path.exists():
        return local_path
    print(">> Downloading Approved Human Gene symbol list...")
    response = requests.get(GENESET_URL, timeout=60)
    response.raise_for_status()
    local_path.write_bytes(response.content)
    return local_path


def load_gene_symbols(catalog_path: Path) -> List[str]:
    symbols: List[str] = []
    with catalog_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            symbol = line.strip()
            if not symbol or symbol.lower().startswith("approved symbol"):
                continue
            if symbol not in symbols:
                symbols.append(symbol)
    if len(symbols) < 50:
        raise RuntimeError("Gene catalog too small after parsing.")
    return symbols


def sample_genes(symbols: List[str], count: int, seed: int | None = 42) -> List[str]:
    rng = random.Random(seed)
    return rng.sample(symbols, count)


def build_prompt(genes: List[str]) -> List[Dict[str, str]]:
    gene_list = ", ".join(genes)
    json_schema = {
        "genes": [
            {
                "symbol": "TP53",
                "diseases": {
                    "cancer": {"associated": True, "evidence": "BRIEF RATIONALE"},
                    "heart_disease": {"associated": False, "evidence": "Why not"},
                    "diabetes": {"associated": False, "evidence": "Why not"},
                    "dementia": {"associated": False, "evidence": "Why not"},
                },
                "interactions": {
                    "has_interactions": True,
                    "partners": ["BRCA1"],
                    "evidence": "Note whether interaction is direct, pathway-level, etc.",
                },
            }
        ]
    }
    user_prompt = (
        "You are a biomedical research assistant. Evaluate the following human genes: "
        f"{gene_list}. For each gene, respond **only** with JSON following exactly this schema:\n"
        f"{json.dumps(json_schema, indent=2)}\n\n"
        "Guidelines:\n"
        "- Limit evidence strings to 2 sentences.\n"
        "- Use publicly known high-level biology knowledge (do not fabricate).\n"
        "- If no solid evidence exists, set `associated` to false and explain briefly.\n"
        "- For interactions, only list other genes from the provided list.\n"
        "- Return valid JSON. No markdown, no commentary."
    )
    return [
        {
            "role": "system",
            "content": (
                "You translate biomedical questions into accurate, concise JSON answers. "
                "Do not include markdown fences or prose outside the JSON object."
            ),
        },
        {"role": "user", "content": user_prompt},
    ]


def call_model(model: str, messages: List[Dict[str, str]], timeout: int = 180) -> str:
    token = get_access_token()
    client = OpenAI(
        api_key=token,
        base_url="https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1",
        timeout=timeout,
    )
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
        max_tokens=4000,
    )
    return response.choices[0].message.content.strip()


def extract_json(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```json\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.rstrip("` \n")
        cleaned = cleaned.rstrip("```")
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        try:
            repaired = repair_json(cleaned)
            return json.loads(repaired)
        except Exception as repair_exc:
            raise ValueError(f"Model response was not valid JSON: {exc}\n{text}") from repair_exc


def parse_results(payload: Dict[str, Any], expected_genes: List[str]) -> List[GeneResult]:
    results: List[GeneResult] = []
    genes = payload.get("genes", [])
    if not isinstance(genes, list):
        raise ValueError("JSON missing 'genes' array.")
    for entry in genes:
        try:
            symbol = entry["symbol"]
            diseases = entry["diseases"]
            interactions = entry["interactions"]
            if symbol not in expected_genes:
                continue
            results.append(
                GeneResult(
                    gene=symbol,
                    has_cancer_link=bool(diseases["cancer"]["associated"]),
                    has_heart_disease_link=bool(diseases["heart_disease"]["associated"]),
                    has_diabetes_link=bool(diseases["diabetes"]["associated"]),
                    has_dementia_link=bool(diseases["dementia"]["associated"]),
                    explanation=" ".join(
                        [
                            diseases["cancer"]["evidence"],
                            diseases["heart_disease"]["evidence"],
                            diseases["diabetes"]["evidence"],
                            diseases["dementia"]["evidence"],
                        ]
                    ),
                    interacting_genes=list(interactions.get("partners", []))
                    if interactions.get("has_interactions")
                    else [],
                )
            )
        except KeyError as exc:
            raise ValueError(f"Malformed entry for gene: {entry}") from exc
    missing = set(expected_genes) - {r.gene for r in results}
    if missing:
        raise ValueError(f"Model response missing genes: {sorted(missing)}")
    return results


def summarize(results: List[GeneResult]) -> Dict[str, Any]:
    aggregated = {
        "total_genes": len(results),
        "disease_counts": {
            "cancer": sum(r.has_cancer_link for r in results),
            "heart_disease": sum(r.has_heart_disease_link for r in results),
            "diabetes": sum(r.has_diabetes_link for r in results),
            "dementia": sum(r.has_dementia_link for r in results),
        },
        "genes_with_interactions": [
            {"gene": r.gene, "partners": r.interacting_genes}
            for r in results
            if r.interacting_genes
        ],
    }
    return aggregated


def write_spot_audit(results: List[GeneResult], genes: List[str], output_dir: Path) -> None:
    by_gene = {r.gene: r for r in results}
    lines = ["# Spot Audit", ""]
    for g in genes:
        r = by_gene.get(g)
        if not r:
            lines.append(f"- {g}: not present in this run")
            continue
        lines.append(
            (
                f"- {g}: cancer={r.has_cancer_link}, heart={r.has_heart_disease_link}, "
                f"diabetes={r.has_diabetes_link}, dementia={r.has_dementia_link}"
            )
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "spot_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_outputs(
    genes: List[str],
    raw_batches: List[Dict[str, Any]],
    payloads: List[Dict[str, Any]],
    results: List[GeneResult],
    summary: Dict[str, Any],
    runtime_sec: float,
    model: str,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "selected_genes.json").write_text(
        json.dumps({"genes": genes}, indent=2),
        encoding="utf-8",
    )
    (output_dir / "raw_model_responses.json").write_text(
        json.dumps(raw_batches, indent=2),
        encoding="utf-8",
    )
    (output_dir / "structured_responses.json").write_text(
        json.dumps(payloads, indent=2),
        encoding="utf-8",
    )
    (output_dir / "results.json").write_text(
        json.dumps([result.__dict__ for result in results], indent=2),
        encoding="utf-8",
    )
    report = {
        "model": model,
        "runtime_seconds": runtime_sec,
        "manual_estimate_minutes": MANUAL_MINUTES_PER_GENE * len(genes),
        "summary": summary,
    }
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    md_lines = [
        "# Gene Analysis Summary",
        f"- Model: `{model}`",
        f"- Runtime (sec): {runtime_sec:.2f}",
        f"- Estimated manual time (minutes): {report['manual_estimate_minutes']:.1f}",
        "",
        "## Disease Counts",
        *[f"- **{disease.replace('_', ' ').title()}**: {count}" for disease, count in summary["disease_counts"].items()],
        "",
        "## Genes With Reported Interactions",
    ]
    if summary["genes_with_interactions"]:
        for item in summary["genes_with_interactions"]:
            partners = ", ".join(item["partners"])
            md_lines.append(f"- {item['gene']}: {partners}")
    else:
        md_lines.append("- None reported")

    (output_dir / "summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the gene-disease analysis workflow.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Sophia model identifier.")
    parser.add_argument("--gene-count", type=int, default=50, help="Number of random genes.")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible gene sampling.",
    )
    args = parser.parse_args()

    catalog_path = ensure_gene_catalog()
    symbols = load_gene_symbols(catalog_path)
    genes = sample_genes(symbols, args.gene_count, seed=args.seed)

    batch_size = 10
    runtime_total = 0.0
    raw_batches: List[Dict[str, Any]] = []
    payloads: List[Dict[str, Any]] = []
    aggregated_results: List[GeneResult] = []

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for batch_idx in range(0, len(genes), batch_size):
        batch_genes = genes[batch_idx : batch_idx + batch_size]
        messages = build_prompt(batch_genes)
        start = time.perf_counter()
        raw_text = call_model(args.model, messages)
        runtime_total += time.perf_counter() - start
        raw_batches.append({"batch": batch_idx // batch_size + 1, "genes": batch_genes, "response": raw_text})
        (OUTPUT_DIR / f"raw_batch_{batch_idx // batch_size + 1:02d}.txt").write_text(raw_text, encoding="utf-8")
        payload = extract_json(raw_text)
        payloads.append(payload)
        batch_results = parse_results(payload, batch_genes)
        aggregated_results.extend(batch_results)

    # Sort aggregated results to match the original gene order
    results_by_gene = {result.gene: result for result in aggregated_results}
    ordered_results = [results_by_gene[g] for g in genes]

    summary = summarize(ordered_results)
    save_outputs(
        genes,
        raw_batches,
        payloads,
        ordered_results,
        summary,
        runtime_total,
        args.model,
        OUTPUT_DIR,
    )

    # Write a simple spot-audit for canonical genes if present
    write_spot_audit(ordered_results, ["TP53", "BRCA1"], OUTPUT_DIR)

    print(f">> Completed gene analysis with {args.model} in {runtime_total:.2f} seconds.")
    print(f">> Outputs written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
