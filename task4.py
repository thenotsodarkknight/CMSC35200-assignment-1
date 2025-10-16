#!/usr/bin/env python3
"""
Task 4: Run the gene-disease analysis locally using Ollama-served models.

This script reuses the sampling and reporting logic from task3.py but swaps the
backend for a local Ollama instance. It measures runtime and stores outputs in
`outputs/gene_analysis_local/<model>/`.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import requests

import task3


def coerce_results(payload: Dict[str, Any], expected_genes: List[str]) -> List[task3.GeneResult]:
    results: Dict[str, task3.GeneResult] = {}
    for entry in payload.get("genes", []):
        try:
            symbol = entry["symbol"]
            if symbol not in expected_genes:
                continue
            diseases = entry["diseases"]
            interactions = entry["interactions"]
            results[symbol] = task3.GeneResult(
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
        except Exception:
            continue

    for gene in expected_genes:
        if gene not in results:
            results[gene] = task3.GeneResult(
                gene=gene,
                has_cancer_link=False,
                has_heart_disease_link=False,
                has_diabetes_link=False,
                has_dementia_link=False,
                explanation="Model omitted this gene in the JSON output.",
                interacting_genes=[],
            )
    return list(results.values())

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_BASE = PROJECT_ROOT / "outputs" / "gene_analysis_local"
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"


def call_ollama(model: str, messages: List[Dict[str, str]], timeout: int = 240) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "format": "json",
        "stream": False,
    }
    response = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    if "message" in data and "content" in data["message"]:
        return data["message"]["content"]
    if "response" in data:
        return data["response"]
    raise ValueError(f"Unexpected Ollama response payload: {json.dumps(data)[:200]}")


def sanitize_model_name(model: str) -> str:
    return model.replace("/", "_").replace(":", "_")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Execute the gene analysis workflow against a local Ollama model."
    )
    parser.add_argument("--model", default="llama3.2:3b", help="Ollama model tag to use.")
    parser.add_argument(
        "--gene-count",
        type=int,
        default=50,
        help="Number of random genes to analyze.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Genes per Ollama request (smaller batches improve JSON compliance).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    catalog_path = task3.ensure_gene_catalog()
    symbols = task3.load_gene_symbols(catalog_path)
    genes = task3.sample_genes(symbols, args.gene_count, seed=args.seed)

    runtime_total = 0.0
    raw_batches: List[Dict[str, Any]] = []
    payloads: List[Dict[str, Any]] = []
    aggregated_results: List[task3.GeneResult] = []

    model_dir = OUTPUT_BASE / sanitize_model_name(args.model)
    model_dir.mkdir(parents=True, exist_ok=True)

    for batch_idx in range(0, len(genes), args.batch_size):
        batch_genes = genes[batch_idx : batch_idx + args.batch_size]
        messages = task3.build_prompt(batch_genes)
        start = time.perf_counter()
        raw_text = call_ollama(args.model, messages)
        runtime_total += time.perf_counter() - start
        raw_batches.append(
            {"batch": batch_idx // args.batch_size + 1, "genes": batch_genes, "response": raw_text}
        )
        (model_dir / f"raw_batch_{batch_idx // args.batch_size + 1:02d}.txt").write_text(
            raw_text, encoding="utf-8"
        )
        payload = task3.extract_json(raw_text)
        payloads.append(payload)
        batch_results = coerce_results(payload, batch_genes)
        aggregated_results.extend(batch_results)

    results_by_gene = {result.gene: result for result in aggregated_results}
    ordered_results = [results_by_gene[g] for g in genes]

    summary = task3.summarize(ordered_results)
    task3.save_outputs(
        genes,
        raw_batches,
        payloads,
        ordered_results,
        summary,
        runtime_total,
        args.model,
        model_dir,
    )

    print(f">> Local analysis with {args.model} complete in {runtime_total:.2f} seconds.")
    print(f">> Outputs written to {model_dir}")


if __name__ == "__main__":
    main()
