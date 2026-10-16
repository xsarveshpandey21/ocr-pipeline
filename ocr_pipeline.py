#!/usr/bin/env python3
"""
Production-quality, CPU-only OCR pipeline optimized for badly degraded historical newspaper pages.

Features:
- Pure CPU execution with multiprocessing (spawn context) and OMP_THREAD_LIMIT=1
- Memory-safe page streaming (never holds all rendered pages in RAM)
- Horizontal projection-profile deskewing (-4 to +4 degrees)
- Median division background flattening to eliminate yellowing, shadows, and scan gradients
- Morphological ruling-line removal to prevent column corruption
- Cleanup variants (raw, light, heavy, auto-band test selection)
- Vertical projection-profile column detection with gutter merging
- Column-wise OCR with Tesseract 5 LSTM engine (--oem 1 --psm 4)
- Relative line-height headline detection (is_headline if height > 1.6x body median)
- Column-major reading order assembly and unique line_ids
- Low-confidence word flagging and optional non-destructive English spell correction
- Keyword relevance ranking with headline boost (1.5x)
- Error isolation across pages and batch output exports (CSV, JSON, TXT, preview JPEGs)
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import multiprocessing as mp
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import pytesseract
from pytesseract import Output

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(processName)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ocr_pipeline")


def configure_environment(
    tesseract_cmd: Optional[str] = None,
    poppler_path: Optional[str] = None,
) -> Tuple[str, Optional[str]]:
    """
    Auto-detect and configure Tesseract and Poppler paths across Windows, macOS, and Linux.
    Honors TESSERACT_CMD and POPPLER_PATH environment variables.

    Returns:
        Tuple of (configured_tesseract_cmd, configured_poppler_path)
    """
    # 1. Tesseract binary discovery
    user_tess = tesseract_cmd or os.environ.get("TESSERACT_CMD")
    if user_tess and os.path.exists(user_tess):
        pytesseract.pytesseract.tesseract_cmd = user_tess
    else:
        # Check standard PATH
        which_tess = shutil.which("tesseract")
        if which_tess:
            pytesseract.pytesseract.tesseract_cmd = which_tess
        else:
            # Common locations
            home = os.path.expanduser("~")
            candidates = [
                os.path.join(home, "scoop", "shims", "tesseract.exe"),
                os.path.join(home, "scoop", "apps", "tesseract", "current", "tesseract.exe"),
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                os.path.join(home, "AppData", "Local", "Programs", "Tesseract-OCR", "tesseract.exe"),
                "/usr/bin/tesseract",
                "/usr/local/bin/tesseract",
                "/opt/homebrew/bin/tesseract",
            ]
            for c in candidates:
                if os.path.exists(c):
                    pytesseract.pytesseract.tesseract_cmd = c
                    break

    # Configure TESSDATA_PREFIX if not already valid
    tess_prefix = os.environ.get("TESSDATA_PREFIX")
    if not tess_prefix or not os.path.exists(tess_prefix):
        tess_bin = pytesseract.pytesseract.tesseract_cmd
        if tess_bin and os.path.exists(tess_bin):
            bin_dir = os.path.dirname(os.path.abspath(tess_bin))
            possible_tessdata = [
                os.path.join(bin_dir, "tessdata"),
                os.path.join(os.path.dirname(bin_dir), "tessdata"),
                os.path.join(os.path.expanduser("~"), "scoop", "apps", "tesseract", "current", "tessdata"),
            ]
            for td in possible_tessdata:
                if os.path.exists(td):
                    os.environ["TESSDATA_PREFIX"] = td
                    break

    # 2. Poppler discovery for pdf2image
    found_poppler = poppler_path or os.environ.get("POPPLER_PATH")
    if not found_poppler or not os.path.exists(found_poppler):
        which_ppm = shutil.which("pdftoppm")
        if which_ppm:
            found_poppler = os.path.dirname(os.path.abspath(which_ppm))
        else:
            home = os.path.expanduser("~")
            poppler_candidates = [
                os.path.join(home, "scoop", "shims"),
                os.path.join(home, "scoop", "apps", "poppler", "current", "bin"),
                r"C:\Program Files\poppler\bin",
                r"C:\poppler\bin",
                "/usr/bin",
                "/usr/local/bin",
                "/opt/homebrew/bin",
            ]
            for pc in poppler_candidates:
                if os.path.exists(pc) and (
                    os.path.exists(os.path.join(pc, "pdftoppm.exe"))
                    or os.path.exists(os.path.join(pc, "pdftoppm"))
                ):
                    found_poppler = pc
                    break

    return pytesseract.pytesseract.tesseract_cmd, found_poppler


# Run configuration on module import
configure_environment()


def get_available_languages() -> List[str]:
    """Retrieve list of installed Tesseract language packs."""
    try:
        langs = pytesseract.get_languages(config="")
        return [l for l in langs if l != "osd"]
    except Exception as e:
        logger.warning(f"Failed to query Tesseract languages: {e}")
        return ["eng"]


def get_pdf_page_count(pdf_path: str) -> int:
    """Return total number of pages in a PDF file safely without loading entire document."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        count = len(doc)
        doc.close()
        return count
    except Exception:
        pass

    try:
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(pdf_path)
        count = len(pdf)
        pdf.close()
        return count
    except Exception:
        pass

    try:
        from pdf2image.pdf2image import pdfinfo_from_path
        _, poppler_path = configure_environment()
        info = pdfinfo_from_path(pdf_path, poppler_path=poppler_path)
        return int(info.get("Pages", 1))
    except Exception as e:
        logger.warning(f"Could not determine PDF page count for {pdf_path}: {e}")
        return 1


def load_page(
    file_path: Union[str, Path],
    page_num: int = 1,
    dpi: int = 300,
    max_dimension: int = 4000,
) -> Tuple[np.ndarray, int, int]:
    """
    Load an image or a specific PDF page as grayscale.
    Downscales proportionately if max(w, h) > max_dimension to preserve CPU memory.

    Args:
        file_path: Path to the PDF or image file.
        page_num: 1-indexed page number (for PDFs).
        dpi: Target rendering resolution for PDFs.
        max_dimension: Maximum pixel length for width or height.

    Returns:
        Tuple of (grayscale_numpy_image, original_width, original_height)
    """
    file_path_str = str(file_path)
    ext = os.path.splitext(file_path_str)[1].lower()

    img_gray: Optional[np.ndarray] = None

    if ext == ".pdf":
        # 1. Try PyMuPDF (fitz) - ultra fast CPU rendering without Poppler dependency
        try:
            import fitz
            doc = fitz.open(file_path_str)
            if page_num <= len(doc):
                page = doc[page_num - 1]
                zoom = dpi / 72.0
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
                img_gray = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w)
            doc.close()
        except Exception:
            img_gray = None

        # 2. Try pypdfium2
        if img_gray is None:
            try:
                import pypdfium2 as pdfium
                pdf = pdfium.PdfDocument(file_path_str)
                if page_num <= len(pdf):
                    scale = dpi / 72.0
                    page = pdf[page_num - 1]
                    pil_img = page.render(scale=scale).to_pil().convert("L")
                    img_gray = np.array(pil_img, dtype=np.uint8)
                pdf.close()
            except Exception:
                img_gray = None

        # 3. Try pdf2image with poppler
        if img_gray is None:
            from pdf2image import convert_from_path
            _, poppler_path = configure_environment()
            images = convert_from_path(
                file_path_str,
                dpi=dpi,
                first_page=page_num,
                last_page=page_num,
                poppler_path=poppler_path,
                grayscale=True,
            )
            if images:
                img_gray = np.array(images[0], dtype=np.uint8)

        if img_gray is None:
            raise RuntimeError(f"Failed to render page {page_num} of PDF: {file_path_str}")

    else:
        # Load regular image (PNG, JPG, TIFF, etc.)
        img = cv2.imread(file_path_str, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Fallback to PIL
            pil_img = Image.open(file_path_str).convert("L")
            img_gray = np.array(pil_img, dtype=np.uint8)
        else:
            img_gray = img

    orig_h, orig_w = img_gray.shape[:2]

    # Dimension clamping
    max_dim = max(orig_h, orig_w)
    if max_dim > max_dimension:
        scale = max_dimension / float(max_dim)
        new_w = max(1, int(round(orig_w * scale)))
        new_h = max(1, int(round(orig_h * scale)))
        img_gray = cv2.resize(img_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return img_gray, orig_w, orig_h


def deskew_page(
    image: np.ndarray,
    max_angle: float = 4.0,
    angle_step: float = 0.25,
) -> Tuple[np.ndarray, float]:
    """
    Estimate skew in [-max_angle, +max_angle] using horizontal projection profile variance.
    Rotates full image with white border fill.

    Args:
        image: Grayscale image.
        max_angle: Search bound in degrees.
        angle_step: Step size in degrees.

    Returns:
        Tuple of (deskewed_image, detected_angle)
    """
    h, w = image.shape[:2]
    # Downscale copy for fast estimation
    target_h = min(h, 1000)
    scale = target_h / float(h)
    small_w = max(1, int(round(w * scale)))
    small = cv2.resize(image, (small_w, target_h), interpolation=cv2.INTER_AREA)

    # Invert binary ink: ink=255, background=0
    _, binary = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    angles = np.arange(-max_angle, max_angle + angle_step, angle_step)
    best_score = -1.0
    best_angle = 0.0

    center = (small_w / 2.0, target_h / 2.0)

    for angle in angles:
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            binary,
            rot_mat,
            (small_w, target_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        # Compute horizontal projection profile (sum of ink along rows)
        row_sums = np.sum(rotated > 0, axis=1, dtype=np.float32)
        score = float(np.var(row_sums))

        if score > best_score:
            best_score = score
            best_angle = float(angle)

    if abs(best_angle) < 0.1:
        return image, 0.0

    # Apply rotation to full-sized image with white border
    full_center = (w / 2.0, h / 2.0)
    rot_mat_full = cv2.getRotationMatrix2D(full_center, best_angle, 1.0)
    deskewed = cv2.warpAffine(
        image,
        rot_mat_full,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )
    # The document's skew angle is -best_angle (since rotating by best_angle straightened it)
    return deskewed, -best_angle


def flatten_background(image: np.ndarray, blur_kernel_size: int = 51) -> np.ndarray:
    """
    Remove uneven illumination, yellowing, and scanner gradient shadows via median division.

    Args:
        image: Grayscale image.
        blur_kernel_size: Kernel dimension for background estimation.

    Returns:
        Illumination-flattened grayscale image.
    """
    h, w = image.shape[:2]
    # Downscale for high-speed median filter
    ds_factor = 4
    small_w = max(1, w // ds_factor)
    small_h = max(1, h // ds_factor)
    small = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_AREA)

    ksize = blur_kernel_size if blur_kernel_size % 2 == 1 else blur_kernel_size + 1
    # Limit kernel size relative to small dimension
    ksize = min(ksize, (min(small_w, small_h) // 2) * 2 - 1)
    ksize = max(ksize, 3)

    bg_small = cv2.medianBlur(small, ksize)
    bg = cv2.resize(bg_small, (w, h), interpolation=cv2.INTER_LINEAR)

    # Division normalization: img / bg * 255
    img_f = image.astype(np.float32)
    bg_f = np.maximum(bg.astype(np.float32), 1.0)
    flattened = np.clip((img_f / bg_f) * 255.0, 0, 255).astype(np.uint8)
    return flattened


def remove_ruling_lines(
    image: np.ndarray,
    min_line_length: int = 40,
    line_thickness: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect long vertical and horizontal ruling lines with morphological opening
    and paint them white to prevent garbage characters and column missegmentation.

    Args:
        image: Grayscale image.
        min_line_length: Minimum pixel length to qualify as a ruling line.
        line_thickness: Dilation thickness for clean removal.

    Returns:
        Tuple of (line_removed_image, binary_lines_mask)
    """
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Horizontal lines
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_line_length, 1))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)

    # Vertical lines
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_line_length))
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

    lines_mask = cv2.bitwise_or(h_lines, v_lines)

    if line_thickness > 1:
        dilate_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (line_thickness, line_thickness)
        )
        lines_mask = cv2.dilate(lines_mask, dilate_kernel)

    cleaned = image.copy()
    cleaned[lines_mask > 0] = 255
    return cleaned, lines_mask


def cleanup_variants(image: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Generate the 3 core cleanup variants:
    - raw: Flattened grayscale.
    - light: Gaussian blur + Otsu thresholding.
    - heavy: Fast bilateral filter + adaptive thresholding + median filter.

    Args:
        image: Grayscale image (already background-flattened and ruling-line removed).

    Returns:
        Dict mapping variant name to processed image.
    """
    # 1. raw: clean grayscale
    raw_var = image.copy()

    # 2. light: light blur + Otsu binarization
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    _, light_var = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3. heavy: bilateral filter + adaptive Gaussian threshold + median denoise
    # Bilateral smoothing keeps sharp character edges while destroying grain
    bilat = cv2.bilateralFilter(image, d=5, sigmaColor=40, sigmaSpace=40)
    heavy_thresh = cv2.adaptiveThreshold(
        bilat,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=25,
        C=10,
    )
    heavy_var = cv2.medianBlur(heavy_thresh, 3)

    return {
        "raw": raw_var,
        "light": light_var,
        "heavy": heavy_var,
    }


def select_best_variant(
    variants: Dict[str, np.ndarray],
    lang: str = "eng",
    sample_band_height_pct: float = 0.25,
) -> Tuple[str, np.ndarray]:
    """
    In 'auto' mode, test-OCR a sample horizontal band across raw, light, and heavy variants.
    Selects the variant yielding the maximum characters with confidence >= 70.

    Args:
        variants: Dict containing raw, light, heavy processed images.
        lang: Tesseract language string.
        sample_band_height_pct: Fraction of page height to sample from the middle.

    Returns:
        Tuple of (winning_variant_name, processed_image)
    """
    ref_img = variants["light"]
    h, w = ref_img.shape[:2]

    # Extract middle horizontal band
    band_h = int(h * sample_band_height_pct)
    y_start = (h - band_h) // 2
    y_end = y_start + band_h

    scores: Dict[str, int] = {}
    config = "--oem 1 --psm 6"

    for name, img in variants.items():
        band = img[y_start:y_end, :]
        try:
            data = pytesseract.image_to_data(
                band, lang=lang, config=config, output_type=Output.DATAFRAME
            )
            data = data.dropna(subset=["text"])
            data["text"] = data["text"].astype(str).str.strip()
            data = data[data["text"] != ""]
            data["conf"] = pd.to_numeric(data["conf"], errors="coerce").fillna(0)

            high_conf_chars = sum(
                len(row["text"]) for _, row in data.iterrows() if row["conf"] >= 70
            )
            scores[name] = high_conf_chars
        except Exception as e:
            logger.debug(f"Error test-OCR on variant {name}: {e}")
            scores[name] = 0

    # Choose winner (break ties towards light, then raw, then heavy)
    preference = ["light", "raw", "heavy"]
    best_name = max(
        scores.keys(),
        key=lambda k: (scores[k], -preference.index(k) if k in preference else 0),
    )
    logger.info(f"Auto-cleanup variant scores: {scores} -> Selected: {best_name}")
    return best_name, variants[best_name]


def detect_columns(
    line_free_mask: np.ndarray,
    min_gutter_width: int = 15,
    min_col_width_pct: float = 0.10,
    gutter_threshold_ratio: float = 0.25,
) -> List[Tuple[int, int, int, int]]:
    """
    Detect column boundaries using smoothed vertical projection profile with
    peak-to-valley gutter detection on the line-free ink mask.
    Falls back to single whole-page column if fewer than 2 distinct columns are found.

    Args:
        line_free_mask: Binary image where ink=255, background=0 (lines already removed).
        min_gutter_width: Minimum pixel width of whitespace to count as a gutter.
        min_col_width_pct: Minimum width as percentage of page width for a valid column.
        gutter_threshold_ratio: Factor above valley floor to qualify as gutter.

    Returns:
        List of column bounding boxes [(x, y, w, h), ...] sorted from left to right.
    """
    h, w = line_free_mask.shape[:2]
    # Sum ink along each vertical column (width dimension)
    vert_proj = np.sum(line_free_mask > 0, axis=0, dtype=np.float32)

    # Smooth projection with 1D Gaussian kernel to avoid intra-word whitespace false gutters
    kernel_size = max(11, (min_gutter_width // 2) * 2 + 1)
    kernel = cv2.getGaussianKernel(kernel_size, sigma=kernel_size / 3.0)
    smoothed = cv2.filter2D(vert_proj, -1, kernel).flatten()

    margin_margin = int(w * 0.05)
    interior = smoothed[margin_margin : w - margin_margin]
    if len(interior) == 0:
        return [(0, 0, w, h)]

    valley_floor = float(np.percentile(interior, 5))
    peak = float(np.percentile(interior, 95))

    # Gutter threshold adapts to valley floor (accounting for full-width headlines and banners)
    gutter_threshold = valley_floor + gutter_threshold_ratio * max(1.0, peak - valley_floor)

    is_gutter = smoothed <= gutter_threshold

    # Find contiguous gutter regions
    gutters: List[Tuple[int, int]] = []
    in_gutter = False
    g_start = 0

    for x in range(w):
        if is_gutter[x]:
            if not in_gutter:
                in_gutter = True
                g_start = x
        else:
            if in_gutter:
                in_gutter = False
                g_end = x - 1
                if (g_end - g_start + 1) >= min_gutter_width:
                    gutters.append((g_start, g_end))
    if in_gutter and (w - 1 - g_start + 1) >= min_gutter_width:
        gutters.append((g_start, w - 1))

    # Filter out page margins (gutters too close to the left or right outer edges)
    interior_gutters = [
        (gs, ge) for gs, ge in gutters if gs > margin_margin and ge < (w - margin_margin)
    ]

    # Split page by interior gutters
    col_bounds: List[Tuple[int, int]] = []
    current_left = 0
    for gs, ge in interior_gutters:
        mid_gutter = (gs + ge) // 2
        col_bounds.append((current_left, mid_gutter))
        current_left = mid_gutter
    col_bounds.append((current_left, w))

    min_col_w = int(w * min_col_width_pct)
    valid_cols = []
    for x1, x2 in col_bounds:
        col_w = x2 - x1
        if col_w >= min_col_w:
            valid_cols.append((x1, 0, col_w, h))

    # Fallback to single page if fewer than 2 columns
    if len(valid_cols) < 2:
        return [(0, 0, w, h)]

    return valid_cols


def ocr_columns(
    image: np.ndarray,
    columns: List[Tuple[int, int, int, int]],
    lang: str = "eng",
    psm: int = 4,
    oem: int = 1,
) -> pd.DataFrame:
    """
    Run Tesseract OCR on each column crop, translating coordinates back to full-page space.

    Args:
        image: Processed image to OCR.
        columns: List of (x, y, w, h) column bounding boxes.
        lang: Tesseract language parameter.
        psm: Page segmentation mode (default 4 = single column of text).
        oem: OCR engine mode (default 1 = LSTM).

    Returns:
        DataFrame with OCR tokens and normalized page coordinates.
    """
    h_img, w_img = image.shape[:2]
    all_rows: List[dict] = []

    # If only 1 column, use psm 3 (automatic segmentation) unless caller forced psm 4
    effective_psm = 3 if len(columns) == 1 and psm == 4 else psm
    config = f"--oem {oem} --psm {effective_psm}"

    pad = 12  # Padding around crops

    for col_idx, (col_x, col_y, col_w, col_h) in enumerate(columns, start=1):
        x1 = max(0, col_x - pad)
        y1 = max(0, col_y - pad)
        x2 = min(w_img, col_x + col_w + pad)
        y2 = min(h_img, col_y + col_h + pad)

        crop = image[y1:y2, x1:x2]

        try:
            data = pytesseract.image_to_data(
                crop, lang=lang, config=config, output_type=Output.DATAFRAME
            )
        except Exception as e:
            logger.warning(f"Error in OCR for column {col_idx}: {e}")
            continue

        if data is None or data.empty:
            continue

        data = data.dropna(subset=["text"])
        data["text"] = data["text"].astype(str).str.strip()
        data = data[data["text"] != ""]

        for _, row in data.iterrows():
            conf_val = float(row.get("conf", 0.0))
            if conf_val < 0:
                conf_val = 0.0

            # Translate back to full-page coordinates
            box_x = int(row["left"]) + x1
            box_y = int(row["top"]) + y1
            box_w = int(row["width"])
            box_h = int(row["height"])

            all_rows.append(
                {
                    "column": col_idx,
                    "block": int(row.get("block_num", 0)),
                    "par": int(row.get("par_num", 0)),
                    "line": int(row.get("line_num", 0)),
                    "word_num": int(row.get("word_num", 0)),
                    "text": str(row["text"]),
                    "conf": conf_val,
                    "x": box_x,
                    "y": box_y,
                    "w": box_w,
                    "h": box_h,
                }
            )

    return pd.DataFrame(all_rows)


def detect_headlines(
    words_df: pd.DataFrame,
    height_multiplier: float = 1.6,
) -> pd.DataFrame:
    """
    Classify lines as headlines: lines whose median word height is > 1.6x the page's
    median body-line height are marked is_headline=True.

    Args:
        words_df: Words dataframe.
        height_multiplier: Threshold factor over body line height.

    Returns:
        DataFrame with 'is_headline' boolean column.
    """
    if words_df.empty:
        words_df["is_headline"] = False
        return words_df

    # Calculate median word height per line
    line_group = words_df.groupby(["column", "block", "par", "line"])["h"].median()
    page_median_height = float(line_group.median()) if not line_group.empty else 16.0
    threshold_height = max(page_median_height * height_multiplier, page_median_height + 4.0)

    # Map back to lines
    is_hl_map = (line_group >= threshold_height).to_dict()

    words_df["is_headline"] = words_df.apply(
        lambda r: is_hl_map.get((r["column"], r["block"], r["par"], r["line"]), False),
        axis=1,
    )
    return words_df


def assign_reading_order(words_df: pd.DataFrame, page_num: int = 1) -> pd.DataFrame:
    """
    Assign column-major global reading order: column 1 top-to-bottom, then column 2, etc.
    Generates unique line_id for each line.

    Args:
        words_df: Words dataframe.
        page_num: Current page number.

    Returns:
        DataFrame sorted with 'order' and 'line_id' columns.
    """
    if words_df.empty:
        for c in ["order", "line_id"]:
            words_df[c] = []
        return words_df

    # Sort columns left-to-right, lines top-to-bottom, words left-to-right
    sorted_df = words_df.sort_values(
        by=["column", "y", "line", "x"], ascending=[True, True, True, True]
    ).copy()

    # Create stable unique line_id per (column, block, par, line)
    line_ids: Dict[Tuple[int, int, int, int], str] = {}
    current_line_idx = 1

    unique_lines = (
        sorted_df[["column", "block", "par", "line", "y"]]
        .drop_duplicates(subset=["column", "block", "par", "line"])
        .sort_values(by=["column", "y"])
    )

    for _, row in unique_lines.iterrows():
        key = (int(row["column"]), int(row["block"]), int(row["par"]), int(row["line"]))
        line_ids[key] = f"P{page_num:03d}_C{key[0]:02d}_L{current_line_idx:04d}"
        current_line_idx += 1

    sorted_df["line_id"] = sorted_df.apply(
        lambda r: line_ids[(int(r["column"]), int(r["block"]), int(r["par"]), int(r["line"]))],
        axis=1,
    )

    # Assign sequential 1-based order
    sorted_df["order"] = np.arange(1, len(sorted_df) + 1, dtype=int)
    return sorted_df


def handle_low_confidence(
    words_df: pd.DataFrame,
    threshold: float = 60.0,
    spell_fix: bool = False,
    lang: str = "eng",
) -> pd.DataFrame:
    """
    Flag words with confidence < threshold as low_conf=True.
    Optionally run pyspellchecker ONLY on low_conf words for English,
    storing corrections in 'text_fixed' while keeping 'text' completely immutable.

    Args:
        words_df: Words dataframe.
        threshold: Confidence cutoff (0-100).
        spell_fix: Enable spell correction.
        lang: Selected language.

    Returns:
        Updated DataFrame with 'low_conf' and 'text_fixed' columns.
    """
    if words_df.empty:
        words_df["low_conf"] = False
        words_df["text_fixed"] = ""
        return words_df

    words_df["low_conf"] = words_df["conf"] < threshold
    words_df["text_fixed"] = words_df["text"].copy()

    if spell_fix and "eng" in lang.lower():
        try:
            from spellchecker import SpellChecker

            spell = SpellChecker()
            # Find unique low-confidence alphabetic words to correct
            low_conf_mask = words_df["low_conf"]
            candidates = set(
                words_df.loc[low_conf_mask, "text"].dropna().astype(str).tolist()
            )

            replacements: Dict[str, str] = {}
            for w in candidates:
                clean_w = "".join(ch for ch in w if ch.isalpha())
                if len(clean_w) >= 3:
                    corr = spell.correction(clean_w.lower())
                    if corr and corr.lower() != clean_w.lower():
                        replacements[w] = corr

            if replacements:
                words_df.loc[low_conf_mask, "text_fixed"] = words_df.loc[
                    low_conf_mask, "text"
                ].map(lambda t: replacements.get(t, t))
        except Exception as e:
            logger.warning(f"Spellchecker encountered an error: {e}")

    return words_df


def rank_keywords(
    words_df: pd.DataFrame,
    keywords: List[str],
) -> pd.DataFrame:
    """
    Rank lines by keyword occurrences * mean confidence, with a 1.5x boost for headlines.

    Args:
        words_df: Words dataframe.
        keywords: List of target keyword strings.

    Returns:
        DataFrame of ranked keyword matches.
    """
    if words_df.empty or not keywords:
        return pd.DataFrame(
            columns=["keyword", "line_id", "column", "is_headline", "score", "line_text", "conf"]
        )

    clean_keywords = [k.strip().lower() for k in keywords if k.strip()]
    if not clean_keywords:
        return pd.DataFrame()

    # Group words into lines
    lines = []
    for line_id, grp in words_df.groupby("line_id"):
        line_text = " ".join(grp["text"].astype(str))
        mean_conf = float(grp["conf"].mean())
        is_hl = bool(grp["is_headline"].iloc[0])
        col = int(grp["column"].iloc[0])

        lower_text = line_text.lower()
        for kw in clean_keywords:
            count = lower_text.count(kw)
            if count > 0:
                boost = 1.5 if is_hl else 1.0
                score = round(count * (mean_conf / 100.0) * boost, 3)
                lines.append(
                    {
                        "keyword": kw,
                        "line_id": line_id,
                        "column": col,
                        "is_headline": is_hl,
                        "score": score,
                        "line_text": line_text,
                        "conf": round(mean_conf, 1),
                    }
                )

    if not lines:
        return pd.DataFrame(
            columns=["keyword", "line_id", "column", "is_headline", "score", "line_text", "conf"]
        )

    res_df = pd.DataFrame(lines).sort_values(by="score", ascending=False)
    return res_df


def generate_annotated_preview(
    image: np.ndarray,
    words_df: pd.DataFrame,
    columns: List[Tuple[int, int, int, int]],
    output_path: str,
) -> str:
    """
    Save annotated preview JPEG to disk (never held in memory):
    - Word boxes: Green (conf >= 70), Orange (50-69), Red (< 50)
    - Headline boxes: Royal Blue with thick stroke
    - Column dividers: Vertical dashed magenta lines

    Args:
        image: Original or processed image.
        words_df: Words dataframe.
        columns: Detected columns.
        output_path: Output file path.

    Returns:
        output_path
    """
    if len(image.shape) == 2:
        preview = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        preview = image.copy()

    h, w = preview.shape[:2]

    # Draw column boundaries
    if len(columns) > 1:
        for x, y, col_w, col_h in columns:
            # Right edge of column
            edge_x = x + col_w
            if edge_x < w - 5:
                cv2.line(preview, (edge_x, 0), (edge_x, h), (200, 0, 200), 2, cv2.LINE_AA)

    # Draw word boxes
    for _, row in words_df.iterrows():
        bx, by, bw, bh = int(row["x"]), int(row["y"]), int(row["w"]), int(row["h"])
        conf = float(row["conf"])
        is_hl = bool(row["is_headline"])

        if is_hl:
            color = (255, 120, 0)  # Royal Blue / Azure (BGR)
            thick = 2
        elif conf >= 70:
            color = (0, 180, 0)  # Green
            thick = 1
        elif conf >= 50:
            color = (0, 140, 255)  # Orange
            thick = 1
        else:
            color = (0, 0, 220)  # Red
            thick = 1

        cv2.rectangle(preview, (bx, by), (bx + bw, by + bh), color, thick)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cv2.imwrite(output_path, preview, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return output_path


def build_structure_json(words_df: pd.DataFrame) -> dict:
    """Generate nested structure: pages -> columns -> lines -> words."""
    structure: dict = {}
    if words_df.empty:
        return structure

    for page_val, p_grp in words_df.groupby("page"):
        p_key = f"page_{page_val}"
        structure[p_key] = {}

        for col_val, c_grp in p_grp.groupby("column"):
            c_key = f"column_{col_val}"
            structure[p_key][c_key] = {}

            for line_id, l_grp in c_grp.groupby("line_id"):
                is_hl = bool(l_grp["is_headline"].iloc[0])
                words_list = []
                for _, w in l_grp.iterrows():
                    words_list.append(
                        {
                            "order": int(w["order"]),
                            "word_num": int(w["word_num"]),
                            "text": str(w["text"]),
                            "text_fixed": str(w.get("text_fixed", w["text"])),
                            "conf": round(float(w["conf"]), 1),
                            "low_conf": bool(w["low_conf"]),
                            "box": [int(w["x"]), int(w["y"]), int(w["w"]), int(w["h"])],
                        }
                    )
                structure[p_key][c_key][line_id] = {
                    "is_headline": is_hl,
                    "words": words_list,
                }
    return structure


def build_reading_order_text(words_df: pd.DataFrame) -> str:
    """
    Format human-readable text:
    - Page & Column headers
    - Headlines prefixed with '## '
    - Low-confidence words wrapped in <angle_brackets>
    """
    if words_df.empty:
        return ""

    out_lines = []
    for page_val, p_grp in words_df.groupby("page"):
        out_lines.append(f"{'='*50}\nPAGE {page_val}\n{'='*50}\n")

        for col_val, c_grp in p_grp.groupby("column"):
            out_lines.append(f"\n--- COLUMN {col_val} ---\n")

            for _, l_grp in c_grp.groupby("line_id"):
                is_hl = bool(l_grp["is_headline"].iloc[0])
                tokens = []
                for _, w in l_grp.iterrows():
                    token = str(w.get("text_fixed", w["text"]))
                    if bool(w.get("low_conf", False)):
                        token = f"<{token}>"
                    tokens.append(token)

                line_str = " ".join(tokens)
                if is_hl:
                    line_str = f"## {line_str}"
                out_lines.append(line_str)

    return "\n".join(out_lines)


@dataclass
class PageProcessConfig:
    file_path: str
    page_num: int
    output_dir: str
    dpi: int = 300
    variant: str = "auto"
    lang: str = "eng"
    psm: int = 4
    low_conf_thresh: float = 60.0
    spell_fix: bool = False
    do_deskew: bool = True
    do_column_split: bool = True
    keywords: List[str] = field(default_factory=list)


def process_single_page(cfg: PageProcessConfig) -> Tuple[dict, pd.DataFrame, pd.DataFrame]:
    """
    Process a single page end-to-end with total error isolation.
    Executed inside worker process; streams outputs to disk.
    """
    start_time = time.time()
    fname = os.path.basename(cfg.file_path)

    stats: dict = {
        "file": fname,
        "page": cfg.page_num,
        "words": 0,
        "mean_conf": 0.0,
        "low_conf_pct": 0.0,
        "num_columns": 0,
        "chosen_variant": cfg.variant,
        "skew_angle": 0.0,
        "seconds": 0.0,
        "error": "",
    }

    words_df = pd.DataFrame()
    kw_df = pd.DataFrame()

    try:
        # 1. Load page
        gray, orig_w, orig_h = load_page(cfg.file_path, page_num=cfg.page_num, dpi=cfg.dpi)

        # 2. Deskew
        skew_angle = 0.0
        if cfg.do_deskew:
            gray, skew_angle = deskew_page(gray)
        stats["skew_angle"] = round(skew_angle, 2)

        # 3. Background flattening
        flattened = flatten_background(gray)

        # 4. Ruling line removal
        line_free, lines_mask = remove_ruling_lines(flattened)

        # 5. Cleanup variants
        variants = cleanup_variants(line_free)
        if cfg.variant == "auto":
            chosen_name, chosen_img = select_best_variant(variants, lang=cfg.lang)
        else:
            chosen_name = cfg.variant
            chosen_img = variants.get(cfg.variant, variants["light"])
        stats["chosen_variant"] = chosen_name

        # 6. Column detection
        h, w = chosen_img.shape[:2]
        if cfg.do_column_split:
            # Ink mask for columns
            _, ink_mask = cv2.threshold(
                chosen_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
            columns = detect_columns(ink_mask)
        else:
            columns = [(0, 0, w, h)]
        stats["num_columns"] = len(columns)

        # 7. Column OCR
        words_df = ocr_columns(chosen_img, columns, lang=cfg.lang, psm=cfg.psm)

        if not words_df.empty:
            words_df["file"] = fname
            words_df["page"] = cfg.page_num

            # 8. Headline detection
            words_df = detect_headlines(words_df)

            # 9. Reading order
            words_df = assign_reading_order(words_df, page_num=cfg.page_num)

            # 10. Low confidence & spell fix
            words_df = handle_low_confidence(
                words_df,
                threshold=cfg.low_conf_thresh,
                spell_fix=cfg.spell_fix,
                lang=cfg.lang,
            )

            # 11. Keyword ranking
            if cfg.keywords:
                kw_df = rank_keywords(words_df, cfg.keywords)

            # Update stats
            total_words = len(words_df)
            mean_conf = float(words_df["conf"].mean())
            low_conf_cnt = int(words_df["low_conf"].sum())
            low_conf_pct = round((low_conf_cnt / total_words) * 100.0, 2) if total_words > 0 else 0.0

            stats["words"] = total_words
            stats["mean_conf"] = round(mean_conf, 2)
            stats["low_conf_pct"] = low_conf_pct

            # 12. Save preview JPEG
            preview_filename = f"{Path(fname).stem}_p{cfg.page_num:03d}_annotated.jpg"
            preview_path = os.path.join(cfg.output_dir, "previews", preview_filename)
            generate_annotated_preview(gray, words_df, columns, preview_path)
        else:
            # Empty page handling
            preview_filename = f"{Path(fname).stem}_p{cfg.page_num:03d}_annotated.jpg"
            preview_path = os.path.join(cfg.output_dir, "previews", preview_filename)
            generate_annotated_preview(gray, pd.DataFrame(), columns, preview_path)

    except Exception as e:
        logger.error(f"Error processing {fname} page {cfg.page_num}: {e}", exc_info=True)
        stats["error"] = str(e)

    elapsed = round(time.time() - start_time, 2)
    stats["seconds"] = elapsed
    return stats, words_df, kw_df


def _worker_entry(cfg: PageProcessConfig) -> Tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Helper entry point for multiprocessing pool."""
    # Ensure worker has single thread for OpenMP / BLAS to avoid CPU thrashing
    os.environ["OMP_THREAD_LIMIT"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    return process_single_page(cfg)


def process_batch(
    file_paths: List[str],
    output_dir: str = "output",
    lang: str = "eng",
    variant: str = "auto",
    workers: Optional[int] = None,
    dpi: int = 300,
    psm: int = 4,
    low_conf_thresh: float = 60.0,
    spell_fix: bool = False,
    do_deskew: bool = True,
    do_column_split: bool = True,
    keywords: Optional[List[str]] = None,
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, str, str]:
    """
    Execute full batch OCR across files using multiprocessing spawn context.
    Streams pages safely without memory spikes and generates all deliverables.

    Returns:
        Tuple of (words_df, page_stats_df, structure_json_path, reading_order_txt_path)
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "previews"), exist_ok=True)

    if keywords is None:
        keywords = []

    # Enumerate all individual pages across files
    work_items: List[PageProcessConfig] = []
    for f in file_paths:
        if not os.path.exists(f):
            continue
        ext = os.path.splitext(f)[1].lower()
        if ext == ".pdf":
            page_count = get_pdf_page_count(f)
            for p in range(1, page_count + 1):
                work_items.append(
                    PageProcessConfig(
                        file_path=f,
                        page_num=p,
                        output_dir=output_dir,
                        dpi=dpi,
                        variant=variant,
                        lang=lang,
                        psm=psm,
                        low_conf_thresh=low_conf_thresh,
                        spell_fix=spell_fix,
                        do_deskew=do_deskew,
                        do_column_split=do_column_split,
                        keywords=keywords,
                    )
                )
        else:
            work_items.append(
                PageProcessConfig(
                    file_path=f,
                    page_num=1,
                    output_dir=output_dir,
                    dpi=dpi,
                    variant=variant,
                    lang=lang,
                    psm=psm,
                    low_conf_thresh=low_conf_thresh,
                    spell_fix=spell_fix,
                    do_deskew=do_deskew,
                    do_column_split=do_column_split,
                    keywords=keywords,
                )
            )

    total_tasks = len(work_items)
    if total_tasks == 0:
        logger.warning("No valid files or pages to process.")
        return pd.DataFrame(), pd.DataFrame(), "", ""

    cpu_avail = mp.cpu_count()
    pool_size = max(1, min(workers or min(4, cpu_avail), cpu_avail))
    logger.info(
        f"Starting batch of {total_tasks} pages with {pool_size} spawn workers (lang={lang}, variant={variant})"
    )

    all_stats: List[dict] = []
    all_words: List[pd.DataFrame] = []
    all_kws: List[pd.DataFrame] = []

    # Use spawn context
    ctx = mp.get_context("spawn")
    completed = 0

    with ctx.Pool(processes=pool_size) as pool:
        for res in pool.imap_unordered(_worker_entry, work_items):
            stats, w_df, k_df = res
            all_stats.append(stats)
            if not w_df.empty:
                all_words.append(w_df)
            if not k_df.empty:
                all_kws.append(k_df)

            completed += 1
            if progress_cb:
                msg = f"Processed {stats['file']} p{stats['page']} ({stats['words']} words, {stats['seconds']}s)"
                progress_cb(completed, total_tasks, msg)
            logger.info(
                f"[{completed}/{total_tasks}] {stats['file']} p{stats['page']} - "
                f"words={stats['words']} conf={stats['mean_conf']}% err={stats['error'] or 'none'} in {stats['seconds']}s"
            )

    # 1. Page Stats CSV
    stats_df = pd.DataFrame(all_stats)
    stats_path = os.path.join(output_dir, "page_stats.csv")
    stats_df.to_csv(stats_path, index=False, encoding="utf-8")

    # 2. Words CSV (UTF-8 with BOM for Excel Hindi/Urdu compatibility)
    words_csv_path = os.path.join(output_dir, "words.csv")
    schema_cols = [
        "file",
        "page",
        "order",
        "column",
        "block",
        "par",
        "line",
        "line_id",
        "word_num",
        "text",
        "text_fixed",
        "conf",
        "low_conf",
        "is_headline",
        "x",
        "y",
        "w",
        "h",
    ]

    if all_words:
        combined_words = pd.concat(all_words, ignore_index=True)
        # Ensure all schema columns present
        for col in schema_cols:
            if col not in combined_words.columns:
                combined_words[col] = ""
        combined_words = combined_words[schema_cols]
    else:
        combined_words = pd.DataFrame(columns=schema_cols)

    combined_words.to_csv(words_csv_path, index=False, encoding="utf-8-sig")

    # 3. Keyword hits CSV
    if all_kws:
        kw_hits_df = pd.concat(all_kws, ignore_index=True)
        kw_path = os.path.join(output_dir, "keyword_hits.csv")
        kw_hits_df.to_csv(kw_path, index=False, encoding="utf-8-sig")

    # 4. Structure JSON
    struct = build_structure_json(combined_words)
    struct_path = os.path.join(output_dir, "structure.json")
    with open(struct_path, "w", encoding="utf-8") as f:
        json.dump(struct, f, indent=2, ensure_ascii=False)

    # 5. Reading order TXT
    reading_txt = build_reading_order_text(combined_words)
    reading_path = os.path.join(output_dir, "reading_order.txt")
    with open(reading_path, "w", encoding="utf-8") as f:
        f.write(reading_txt)

    logger.info(f"Batch completed! All deliverables saved in: {output_dir}")
    return combined_words, stats_df, struct_path, reading_path


def main():
    """Headless Command-Line Interface."""
    parser = argparse.ArgumentParser(
        description="CPU-only OCR pipeline for degraded historical newspapers."
    )
    parser.add_argument("input_path", help="Path to PDF/image file or directory of pages.")
    parser.add_argument("--output-dir", default="output", help="Directory to save deliverables.")
    parser.add_argument("--lang", default="eng", help="Tesseract language(s), e.g. eng, eng+hin.")
    parser.add_argument(
        "--variant",
        default="auto",
        choices=["auto", "raw", "light", "heavy"],
        help="Image restoration variant.",
    )
    parser.add_argument("--workers", type=int, default=None, help="Number of CPU worker processes.")
    parser.add_argument("--dpi", type=int, default=300, help="PDF rendering DPI.")
    parser.add_argument("--psm", type=int, default=4, help="Tesseract PSM mode.")
    parser.add_argument(
        "--low-conf-thresh", type=float, default=60.0, help="Low confidence cutoff."
    )
    parser.add_argument(
        "--spell-fix", action="store_true", help="Enable non-destructive spell fix on low-conf words."
    )
    parser.add_argument(
        "--no-deskew", action="store_true", help="Disable horizontal projection deskewing."
    )
    parser.add_argument(
        "--no-columns", action="store_true", help="Disable multi-column gutter detection."
    )
    parser.add_argument(
        "--keywords", nargs="*", default=[], help="Keywords to rank for relevancy."
    )

    args = parser.parse_args()

    # Collect files
    in_path = args.input_path
    if os.path.isdir(in_path):
        patterns = ["*.pdf", "*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"]
        files = []
        for p in patterns:
            files.extend(glob.glob(os.path.join(in_path, p)))
            files.extend(glob.glob(os.path.join(in_path, p.upper())))
        files = sorted(list(set(files)))
    elif os.path.isfile(in_path):
        files = [in_path]
    else:
        logger.error(f"Input path not found: {in_path}")
        sys.exit(1)

    if not files:
        logger.error(f"No PDF or image files found in {in_path}")
        sys.exit(1)

    print(f"Discovered {len(files)} files to process.")

    words_df, stats_df, s_path, r_path = process_batch(
        file_paths=files,
        output_dir=args.output_dir,
        lang=args.lang,
        variant=args.variant,
        workers=args.workers,
        dpi=args.dpi,
        psm=args.psm,
        low_conf_thresh=args.low_conf_thresh,
        spell_fix=args.spell_fix,
        do_deskew=not args.no_deskew,
        do_column_split=not args.no_columns,
        keywords=args.keywords,
    )

    print("\n" + "=" * 50)
    print("PAGE PROCESSING SUMMARY")
    print("=" * 50)
    if not stats_df.empty:
        summary_cols = ["file", "page", "words", "mean_conf", "low_conf_pct", "num_columns", "seconds"]
        print(stats_df[[c for c in summary_cols if c in stats_df.columns]].to_string(index=False))
    print("=" * 50)
    print(f"Words CSV: {os.path.join(args.output_dir, 'words.csv')}")
    print(f"Page Stats: {os.path.join(args.output_dir, 'page_stats.csv')}")
    print(f"Reading Order: {r_path}")
    print(f"Structure JSON: {s_path}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
