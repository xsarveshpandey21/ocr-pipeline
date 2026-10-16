#!/usr/bin/env python3
"""
Benchmark and comparison script for the CPU-only newspaper OCR pipeline.
Runs:
1. End-to-end evaluation on synthetic degraded newspaper pages.
2. Comparative analysis of auto vs fixed variants (raw, light, heavy) on the worst-case page.
3. Formats performance, accuracy, and speed metrics.
"""

import json
import os
import time
from pathlib import Path
import pandas as pd

from ocr_pipeline import (
    PageProcessConfig,
    configure_environment,
    process_batch,
    process_single_page,
)


def run_benchmark():
    configure_environment()
    samples_dir = Path("samples")
    sample_files = sorted(list(samples_dir.glob("*.png")))
    assert sample_files, "No sample files found in samples/ directory!"

    print("=" * 70)
    print("1. RUNNING FULL PIPELINE ON ALL SAMPLE PAGES (auto variant, 4 CPU workers)")
    print("=" * 70)

    start_batch = time.time()
    words_df, stats_df, struct_path, reading_path = process_batch(
        file_paths=[str(p) for p in sample_files],
        output_dir="output_benchmark",
        variant="auto",
        workers=min(4, os.cpu_count() or 4),
        keywords=["diplomats", "treaty", "railway", "harvest"],
    )
    total_batch_time = round(time.time() - start_batch, 2)

    print("\n--- SAMPLE PAGES BENCHMARK RESULTS ---")
    summary_cols = ["file", "page", "words", "mean_conf", "low_conf_pct", "num_columns", "chosen_variant", "skew_angle", "seconds"]
    print(stats_df[summary_cols].to_markdown(index=False))
    print(f"\nTotal batch processing time: {total_batch_time}s (Average: {round(total_batch_time / len(sample_files), 2)}s per page)\n")

    # 2. Comparative Variant Analysis on Page 3 (Worst Case)
    worst_case_file = str(samples_dir / "page_3_worst_case.png")
    print("=" * 70)
    print("2. COMPARATIVE VARIANT ANALYSIS ON WORST-CASE PAGE (page_3_worst_case.png)")
    print("=" * 70)

    variants_to_test = ["auto", "raw", "light", "heavy"]
    variant_results = []

    for v in variants_to_test:
        cfg = PageProcessConfig(
            file_path=worst_case_file,
            page_num=1,
            output_dir=f"output_variants/{v}",
            variant=v,
            lang="eng",
            psm=4,
            low_conf_thresh=60.0,
            do_deskew=True,
            do_column_split=True,
        )
        stats, w_df, _ = process_single_page(cfg)

        high_conf_chars = 0
        if not w_df.empty:
            high_conf_chars = sum(len(str(t)) for _, row in w_df.iterrows() if row["conf"] >= 70 for t in [row["text"]])

        variant_results.append({
            "Variant": v,
            "Words Extracted": stats["words"],
            "Mean Confidence (%)": stats["mean_conf"],
            "Low Conf (%)": stats["low_conf_pct"],
            "High-Conf Chars (conf >= 70)": high_conf_chars,
            "Skew Detected (deg)": stats["skew_angle"],
            "Columns Detected": stats["num_columns"],
            "Time (s)": stats["seconds"],
        })

    var_df = pd.DataFrame(variant_results)
    print(var_df.to_markdown(index=False))

    # Determine winner
    # Score based on highest count of high-confidence characters and mean confidence
    var_df["Score"] = var_df["High-Conf Chars (conf >= 70)"] * (var_df["Mean Confidence (%)"] / 100.0)
    winner_row = var_df.sort_values(by="Score", ascending=False).iloc[0]
    winner_name = winner_row["Variant"]

    print("\n" + "=" * 70)
    print(f"WINNER: Variant '{winner_name}'")
    print(f"Rationale: Achieved {winner_row['High-Conf Chars (conf >= 70)']} high-confidence characters with {winner_row['Mean Confidence (%)']}% mean confidence in {winner_row['Time (s)']}s.")
    print("=" * 70)

    # Export benchmark report data as json for the final report
    report_data = {
        "pages_summary": stats_df[summary_cols].to_dict(orient="records"),
        "total_batch_time": total_batch_time,
        "variant_comparison": variant_results,
        "winner": winner_name,
        "winner_details": winner_row.to_dict(),
    }
    with open("benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)


if __name__ == "__main__":
    run_benchmark()
