#!/usr/bin/env python3
"""
Gradio Web Application for CPU-Only Newspaper OCR Pipeline.
Optimized for badly degraded historical newspapers with multi-column layout.

Features:
- Multi-file batch upload (PDF, PNG, JPG, JPEG, TIFF)
- Dynamic Tesseract language selection populated from local installation
- Cleanup variants: auto, raw, light, heavy
- Advanced configuration accordion (DPI, Workers, PSM, Deskew, Column Split, Spell Fix, Low-Conf Threshold)
- Real-time progress bar updating per processed page
- Visual results: Interactive confidence-map gallery with color-coded bounding boxes
- Reading order reader view (headlines in ##, low-conf words in <angle brackets>)
- Searchable words table (capped in UI to 5,000 rows for snappy performance)
- Keyword relevance ranking table with headline boost
- Page performance & execution stats
- Direct file download buttons + Full ZIP package export
- Clear error handling (no files, missing Tesseract, missing Poppler)
- Queue enabled with concurrency limit = 1 and spawn multiprocessing safety
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple

import gradio as gr
import pandas as pd

from ocr_pipeline import (
    configure_environment,
    get_available_languages,
    process_batch,
)

# Custom CSS for modern design aesthetics
CUSTOM_CSS = """
/* Antigravity Premium Newspaper OCR Theme */
:root {
    --primary-color: #3b82f6;
    --accent-color: #8b5cf6;
    --card-bg: rgba(255, 255, 255, 0.05);
    --border-subtle: rgba(255, 255, 255, 0.12);
}

.gradio-container {
    max-width: 1350px !important;
    margin: 0 auto !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

.hero-banner {
    background: linear-gradient(135deg, #1e1b4b 0%, #172554 50%, #0f172a 100%);
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-radius: 16px;
    padding: 24px 32px;
    margin-bottom: 24px;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
}

.hero-title {
    font-size: 2.1rem;
    font-weight: 800;
    letter-spacing: -0.025em;
    background: linear-gradient(to right, #ffffff, #93c5fd, #c4b5fd);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
}

.hero-sub {
    font-size: 1.05rem;
    color: #94a3b8;
    line-height: 1.5;
}

.badge-row {
    display: flex;
    gap: 10px;
    margin-top: 14px;
    flex-wrap: wrap;
}

.sys-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 9999px;
    font-size: 0.8rem;
    color: #e2e8f0;
    font-weight: 500;
}

.sys-badge.green {
    border-color: rgba(34, 197, 94, 0.4);
    background: rgba(34, 197, 94, 0.12);
    color: #86efac;
}

.stat-metric-card {
    border-radius: 12px;
    border: 1px solid var(--border-subtle);
    padding: 14px;
    background: var(--card-bg);
}

.reading-order-box textarea {
    font-family: 'JetBrains Mono', 'Courier New', monospace !important;
    font-size: 0.92rem !important;
    line-height: 1.6 !important;
}
"""


def verify_system_health() -> Tuple[bool, str, bool, str]:
    """Check Tesseract and Poppler status."""
    tess_cmd, poppler_path = configure_environment()
    tess_ok = bool(tess_cmd and os.path.exists(tess_cmd))
    poppler_ok = bool(
        poppler_path
        and (
            os.path.exists(os.path.join(poppler_path, "pdftoppm.exe"))
            or os.path.exists(os.path.join(poppler_path, "pdftoppm"))
            or shutil.which("pdftoppm")
        )
    )
    return tess_ok, tess_cmd or "Not detected", poppler_ok, poppler_path or "Not detected"


def create_zip_archive(output_dir: str) -> str:
    """Create a consolidated zip archive of all outputs in output_dir."""
    zip_path = os.path.join(output_dir, "ocr_bundle.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(output_dir):
            for file in files:
                if file.endswith(".zip"):
                    continue
                full_p = os.path.join(root, file)
                rel_p = os.path.relpath(full_p, output_dir)
                zf.write(full_p, arcname=rel_p)
    return zip_path


def run_ocr_pipeline(
    uploaded_files: Optional[List[gr.FileData]],
    selected_langs: List[str],
    cleanup_variant: str,
    keywords_str: str,
    dpi_val: int,
    psm_val: int,
    low_conf_val: float,
    workers_val: int,
    enable_deskew: bool,
    enable_columns: bool,
    enable_spell: bool,
    progress=gr.Progress(track_tqdm=True),
):
    """Gradio runner function executed upon clicking Run."""
    def empty_response(msg: str):
        return (
            msg,
            [],
            "",
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            None,
            None,
            None,
            None,
            None,
        )

    # 1. Validation checks
    if not uploaded_files:
        gr.Warning("Please upload at least one PDF or image scan before running.")
        return empty_response("### ⚠️ No Files Uploaded\nPlease drag & drop or select scans in the upload panel on the left, then click **Run OCR Batch**.")

    tess_ok, tess_path, poppler_ok, _ = verify_system_health()
    if not tess_ok:
        gr.Warning(f"Tesseract binary not found! Path: {tess_path}")
        return empty_response(f"### ❌ Tesseract Missing\nTesseract binary was not found at `{tess_path}`. Please ensure Tesseract is installed.")

    # Extract file paths safely (handles str, gr.FileData, or file objects)
    file_paths = []
    for f in uploaded_files:
        if isinstance(f, str):
            file_paths.append(f)
        elif hasattr(f, "path") and f.path:
            file_paths.append(str(f.path))
        elif hasattr(f, "name") and f.name:
            file_paths.append(str(f.name))

    if not file_paths:
        gr.Warning("No valid file paths found in upload.")
        return empty_response("### ⚠️ Invalid Files\nNo readable file paths could be found. Please try re-uploading your scans.")

    # Check if any uploaded files are PDFs and if poppler/pymupdf is available
    has_pdf = any(p.lower().endswith(".pdf") for p in file_paths)
    if has_pdf and not poppler_ok:
        try:
            import fitz
        except ImportError:
            gr.Warning("Poppler and PyMuPDF are missing for PDF parsing.")
            return empty_response("### ❌ Missing PDF Support\nPDF uploaded but Poppler is missing and PyMuPDF is not installed.")

    # 2. Setup job output directory
    job_dir = tempfile.mkdtemp(prefix="ocr_job_")

    # Language string
    lang_str = "+".join(selected_langs) if selected_langs else "eng"

    # Keywords list
    kw_list = [k.strip() for k in keywords_str.split(",") if k.strip()] if keywords_str else []

    progress(0.05, desc="Initializing parallel workers...")

    def progress_callback(done: int, total: int, msg: str):
        frac = min(0.95, 0.05 + 0.90 * (done / max(1, total)))
        progress(frac, desc=f"Page {done}/{total}: {msg}")

    # 3. Execute batch processing
    try:
        words_df, stats_df, struct_path, reading_path = process_batch(
            file_paths=file_paths,
            output_dir=job_dir,
            lang=lang_str,
            variant=cleanup_variant,
            workers=int(workers_val),
            dpi=int(dpi_val),
            psm=int(psm_val),
            low_conf_thresh=float(low_conf_val),
            spell_fix=bool(enable_spell),
            do_deskew=bool(enable_deskew),
            do_column_split=bool(enable_columns),
            keywords=kw_list,
            progress_cb=progress_callback,
        )
    except Exception as e:
        gr.Warning(f"OCR Pipeline failed during processing: {e}")
        return empty_response(f"### ❌ Processing Error\nAn error occurred while processing: `{e}`")

    progress(0.96, desc="Packaging deliverables...")

    # 4. Gather deliverables
    previews_dir = os.path.join(job_dir, "previews")
    preview_images = []
    if os.path.exists(previews_dir):
        for img_name in sorted(os.listdir(previews_dir)):
            if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                img_p = os.path.join(previews_dir, img_name)
                preview_images.append((img_p, img_name))

    # Read formatted reading order text
    reading_text = ""
    if os.path.exists(reading_path):
        with open(reading_path, "r", encoding="utf-8") as f:
            reading_text = f.read()

    # Cap words table for UI rendering performance (5000 rows), full dataframe available in download
    display_words = words_df.head(5000) if not words_df.empty else pd.DataFrame()

    # Keyword hits
    kw_path = os.path.join(job_dir, "keyword_hits.csv")
    kw_df = pd.read_csv(kw_path, encoding="utf-8-sig") if os.path.exists(kw_path) else pd.DataFrame()

    # Create ZIP
    zip_bundle_path = create_zip_archive(job_dir)
    zip_bundle_path = zip_bundle_path if (zip_bundle_path and os.path.exists(zip_bundle_path)) else None
    words_csv_path = os.path.join(job_dir, "words.csv") if os.path.exists(os.path.join(job_dir, "words.csv")) else None
    stats_csv_path = os.path.join(job_dir, "page_stats.csv") if os.path.exists(os.path.join(job_dir, "page_stats.csv")) else None
    struct_path = struct_path if (struct_path and os.path.exists(struct_path)) else None
    reading_path = reading_path if (reading_path and os.path.exists(reading_path)) else None

    progress(1.0, desc="Done!")

    # Summary metric markdown
    total_words = len(words_df)
    mean_conf = round(float(words_df["conf"].mean()), 1) if not words_df.empty else 0.0
    low_conf_pct = round(float(words_df["low_conf"].sum()) / max(1, total_words) * 100.0, 1) if not words_df.empty else 0.0
    total_pages = len(stats_df)
    total_seconds = round(float(stats_df["seconds"].sum()), 1) if not stats_df.empty else 0.0

    summary_md = f"""
### ⚡ Execution Summary
* **Total Pages:** `{total_pages}` &nbsp;|&nbsp; **Recognized Words:** `{total_words:,}` &nbsp;|&nbsp; **Overall Mean Confidence:** `{mean_conf}%`
* **Low-Confidence Words (<{low_conf_val}%):** `{low_conf_pct}%` &nbsp;|&nbsp; **Total Processing Time:** `{total_seconds}s`
"""

    return (
        summary_md,
        preview_images,
        reading_text,
        display_words,
        kw_df,
        stats_df,
        zip_bundle_path,
        words_csv_path,
        struct_path,
        reading_path,
        stats_csv_path,
    )


def build_app() -> gr.Blocks:
    """Build the Gradio interface."""
    available_langs = get_available_languages()
    tess_ok, tess_path, poppler_ok, poppler_path = verify_system_health()
    cpu_count = os.cpu_count() or 4

    demo = gr.Blocks(title="Antigravity OCR Pipeline")

    with demo:
        # Hero Header Banner
        gr.HTML(f"""
        <div class="hero-banner">
            <div class="hero-title">📰 OCR Pipeline</div>
            <div class="hero-sub">
                Industrial-grade, high-throughput OCR pipeline engineered for badly degraded documents and multi-column scans.
                Features deskewing, illumination flattening, line erasure, multi-column gutter segmentation,
                reading-order preservation, and headline classification.
            </div>
            <div class="badge-row">
                <span class="sys-badge {'green' if tess_ok else ''}">
                    ⚙️ Tesseract: {'v5 Ready' if tess_ok else 'Missing'}
                </span>
                <span class="sys-badge {'green' if poppler_ok else ''}">
                    📄 Poppler / PDF: {'Configured' if poppler_ok else 'Fallback PyMuPDF'}
                </span>
                <span class="sys-badge">
                    💻 CPU Cores: {cpu_count}
                </span>
                <span class="sys-badge">
                    🌐 Installed OCR Languages: {len(available_langs)}
                </span>
            </div>
        </div>
        """)

        with gr.Row():
            # Left Control Column
            with gr.Column(scale=4):
                upload_input = gr.File(
                    label="Upload Scans (PDF, PNG, JPG, TIFF)",
                    file_count="multiple",
                    file_types=[".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif"],
                )

                with gr.Row():
                    lang_select = gr.Dropdown(
                        label="OCR Language(s)",
                        choices=available_langs,
                        value=["eng"] if "eng" in available_langs else available_langs[:1],
                        multiselect=True,
                        info="Select one or more installed Tesseract languages.",
                    )
                    variant_radio = gr.Radio(
                        label="Cleanup Variant",
                        choices=["auto", "raw", "light", "heavy"],
                        value="auto",
                        info="Auto tests variants on sample band before full page.",
                    )

                keywords_input = gr.Textbox(
                    label="Relevance Keywords (Optional)",
                    placeholder="e.g. parliament, treaty, harvest, railway",
                    info="Comma-separated keywords to score and rank across lines.",
                )

                # Advanced Accordion
                with gr.Accordion("🛠️ Advanced OCR & Preprocessing Parameters", open=False):
                    with gr.Row():
                        workers_slider = gr.Slider(
                            minimum=1,
                            maximum=cpu_count,
                            value=min(4, cpu_count),
                            step=1,
                            label="Parallel CPU Workers",
                            info="Multiprocessing spawn pool size.",
                        )
                        dpi_slider = gr.Slider(
                            minimum=150,
                            maximum=450,
                            value=300,
                            step=50,
                            label="PDF Rendering DPI",
                            info="Resolution for rasterizing PDF pages.",
                        )

                    with gr.Row():
                        low_conf_slider = gr.Slider(
                            minimum=0.0,
                            maximum=100.0,
                            value=60.0,
                            step=5.0,
                            label="Low Confidence Threshold",
                            info="Words under this score are tagged low_conf.",
                        )
                        psm_dropdown = gr.Dropdown(
                            label="Tesseract PSM Mode",
                            choices=[3, 4, 6, 11],
                            value=4,
                            info="4 = Single column of variable text (recommended for multi-column split).",
                        )

                    with gr.Row():
                        deskew_chk = gr.Checkbox(
                            label="Horizontal Projection Deskew",
                            value=True,
                            info="Correct -4° to +4° scan skew.",
                        )
                        col_split_chk = gr.Checkbox(
                            label="Vertical Projection Column Split",
                            value=True,
                            info="Isolate columns before OCR to maintain reading order.",
                        )
                        spell_chk = gr.Checkbox(
                            label="Spell-Fix Low-Conf Words (English)",
                            value=False,
                            info="Non-destructive suggestions in text_fixed column.",
                        )

                run_btn = gr.Button("🚀 Run OCR Batch", variant="primary", size="lg")

            # Right Results Column
            with gr.Column(scale=8):
                summary_output = gr.Markdown(value="*Upload scans and click **Run OCR Batch** to view results.*")

                with gr.Tabs() as tabs:
                    with gr.Tab("🖼️ Confidence Maps"):
                        gallery_output = gr.Gallery(
                            label="Annotated Page Previews (Green ≥70, Orange 50-69, Red <50, Blue=Headlines, Magenta=Columns)",
                            columns=2,
                            height="auto",
                            preview=True,
                            object_fit="contain",
                        )

                    with gr.Tab("📖 Reading Order Reader"):
                        gr.Markdown("*Headlines tagged with `##`, low-confidence words wrapped in `<angle brackets>`.*")
                        reading_output = gr.Textbox(
                            label="Preserved Reading Order Text",
                            lines=20,
                            elem_classes=["reading-order-box"],
                        )

                    with gr.Tab("📋 Words Table"):
                        gr.Markdown("*Displays first 5,000 words for UI responsiveness. Full dataset available in downloads.*")
                        words_df_output = gr.Dataframe(
                            label="Extracted Words Dataframe",
                            headers=[
                                "file", "page", "order", "column", "block", "par", "line",
                                "line_id", "word_num", "text", "text_fixed", "conf",
                                "low_conf", "is_headline", "x", "y", "w", "h"
                            ],
                            wrap=True,
                        )

                    with gr.Tab("🔍 Keyword Relevance"):
                        kw_df_output = gr.Dataframe(
                            label="Ranked Keyword Occurrences (1.5x Headline Boost)",
                            wrap=True,
                        )

                    with gr.Tab("📊 Page Stats"):
                        stats_df_output = gr.Dataframe(
                            label="Per-Page Execution Performance & Confidence",
                            wrap=True,
                        )

                    with gr.Tab("💾 Download Center"):
                        gr.Markdown("### 📥 Download Processed Deliverables")
                        zip_file_output = gr.File(label="📦 Download Complete Bundle (.zip)")
                        with gr.Row():
                            words_csv_file = gr.File(label="📄 words.csv (UTF-8 BOM)")
                            struct_json_file = gr.File(label="📊 structure.json")
                        with gr.Row():
                            reading_txt_file = gr.File(label="📑 reading_order.txt")
                            stats_csv_file = gr.File(label="📈 page_stats.csv")

        # Event Binding
        run_btn.click(
            fn=run_ocr_pipeline,
            inputs=[
                upload_input,
                lang_select,
                variant_radio,
                keywords_input,
                dpi_slider,
                psm_dropdown,
                low_conf_slider,
                workers_slider,
                deskew_chk,
                col_split_chk,
                spell_chk,
            ],
            outputs=[
                summary_output,
                gallery_output,
                reading_output,
                words_df_output,
                kw_df_output,
                stats_df_output,
                zip_file_output,
                words_csv_file,
                struct_json_file,
                reading_txt_file,
                stats_csv_file,
            ],
        )

    return demo


if __name__ == "__main__":
    # Required for multiprocessing spawn safety
    demo = build_app()
    demo.queue(default_concurrency_limit=1)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(),
        css=CUSTOM_CSS,
    )
