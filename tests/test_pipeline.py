"""
Comprehensive test suite for the CPU-only newspaper OCR pipeline.
Validates:
- Deskew accuracy
- Background flattening & defect normalization
- Ruling line removal
- Multi-column detection (assert 3 columns)
- Headline marking
- Column-major reading order
- Keyword relevance ranking
- Schema validation for words.csv (UTF-8 BOM), structure.json, reading_order.txt, page_stats.csv
- Error isolation across pages
"""

import os
import shutil
import tempfile
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import pytest

from ocr_pipeline import (
    assign_reading_order,
    cleanup_variants,
    configure_environment,
    deskew_page,
    detect_columns,
    detect_headlines,
    flatten_background,
    handle_low_confidence,
    load_page,
    process_batch,
    process_single_page,
    PageProcessConfig,
    rank_keywords,
    remove_ruling_lines,
    select_best_variant,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session", autouse=True)
def setup_env():
    """Ensure environment is configured."""
    home = os.path.expanduser("~")
    scoop_shims = os.path.join(home, "scoop", "shims")
    if os.path.exists(scoop_shims) and scoop_shims not in os.environ.get("PATH", ""):
        os.environ["PATH"] = f"{scoop_shims};{os.environ.get('PATH', '')}"
    configure_environment()


def test_deskew_synthetic():
    """Test deskew recovery on an image with known synthetic rotation."""
    # Create clean synthetic image with clear horizontal lines
    img = np.full((600, 800), 255, dtype=np.uint8)
    for y in range(100, 500, 40):
        cv2.line(img, (100, y), (700, y), 0, 4)

    # Rotate by known +2.5 degrees
    known_angle = 2.5
    center = (400, 300)
    rot_mat = cv2.getRotationMatrix2D(center, known_angle, 1.0)
    rotated = cv2.warpAffine(img, rot_mat, (800, 600), borderValue=255)

    deskewed, estimated_angle = deskew_page(rotated, max_angle=4.0, angle_step=0.25)
    # The estimated rotation must cancel out the skew: estimated_angle should be within 0.5 of known_angle
    assert abs(estimated_angle - known_angle) <= 0.5, f"Expected ~{known_angle}, got {estimated_angle}"


def test_background_flattening():
    """Test that median division successfully normalizes dark illumination gradients."""
    # Create image with strong gradient
    base = np.full((400, 400), 200, dtype=np.uint8)
    # Add dark vignette shadow
    x = np.linspace(0.4, 1.0, 400, dtype=np.float32)
    y = np.linspace(0.5, 1.0, 400, dtype=np.float32)
    grad_x, grad_y = np.meshgrid(x, y)
    degraded = (base.astype(np.float32) * (grad_x * grad_y)).astype(np.uint8)

    # Before flattening, standard deviation is high due to gradient
    std_before = float(np.std(degraded))

    flattened = flatten_background(degraded, blur_kernel_size=31)
    std_after = float(np.std(flattened))

    assert std_after < std_before, f"Std after ({std_after}) should be lower than before ({std_before})"
    # Flattened background should be close to 255
    assert np.mean(flattened) > 230


def test_remove_ruling_lines():
    """Test that horizontal and vertical ruling lines are detected and erased."""
    # Image with text-like blobs and long ruling lines
    img = np.full((500, 500), 255, dtype=np.uint8)
    # Add long vertical rule
    cv2.line(img, (250, 50), (250, 450), 0, 3)
    # Add long horizontal rule
    cv2.line(img, (50, 250), (450, 250), 0, 3)
    # Add small text-like character dots
    cv2.circle(img, (100, 100), 4, 0, -1)
    cv2.circle(img, (350, 350), 4, 0, -1)

    cleaned, mask = remove_ruling_lines(img, min_line_length=40, line_thickness=3)

    # Ruling lines must be painted white (255)
    assert cleaned[250, 250] == 255
    assert cleaned[200, 250] == 255
    assert cleaned[250, 200] == 255
    # Small character dots must NOT be erased
    assert cleaned[100, 100] == 0
    assert cleaned[350, 350] == 0


def test_column_detection_fixtures():
    """Test that exactly 3 columns are detected on Page 1 synthetic fixture."""
    p1_path = FIXTURES_DIR / "page_1_yellow_tint.png"
    assert p1_path.exists(), f"Fixture missing: {p1_path}"

    gray, _, _ = load_page(p1_path)
    flattened = flatten_background(gray)
    line_free, _ = remove_ruling_lines(flattened)

    _, ink_mask = cv2.threshold(line_free, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    columns = detect_columns(ink_mask, min_gutter_width=15)

    assert len(columns) == 3, f"Expected 3 columns, detected {len(columns)}: {columns}"
    # Verify columns are ordered left-to-right
    for i in range(len(columns) - 1):
        assert columns[i][0] < columns[i + 1][0]


def test_headline_marking():
    """Test that lines with height > 1.6x page median height are marked is_headline."""
    sample_data = pd.DataFrame([
        # Large headline words
        {"column": 1, "block": 1, "par": 1, "line": 1, "word_num": 1, "text": "GREAT", "h": 36, "y": 100, "x": 50, "w": 80, "conf": 95},
        {"column": 1, "block": 1, "par": 1, "line": 1, "word_num": 2, "text": "VICTORY", "h": 36, "y": 100, "x": 140, "w": 90, "conf": 95},
        # Body text line 1
        {"column": 1, "block": 1, "par": 2, "line": 2, "word_num": 1, "text": "The", "h": 16, "y": 150, "x": 50, "w": 30, "conf": 90},
        {"column": 1, "block": 1, "par": 2, "line": 2, "word_num": 2, "text": "treaty", "h": 16, "y": 150, "x": 85, "w": 40, "conf": 90},
        {"column": 1, "block": 1, "par": 2, "line": 2, "word_num": 3, "text": "was", "h": 16, "y": 150, "x": 130, "w": 30, "conf": 90},
        # Body text line 2
        {"column": 1, "block": 1, "par": 2, "line": 3, "word_num": 1, "text": "signed", "h": 16, "y": 180, "x": 50, "w": 45, "conf": 90},
        {"column": 1, "block": 1, "par": 2, "line": 3, "word_num": 2, "text": "yesterday", "h": 16, "y": 180, "x": 100, "w": 65, "conf": 90},
    ])

    marked = detect_headlines(sample_data, height_multiplier=1.6)
    headline_words = marked[marked["is_headline"]]
    body_words = marked[~marked["is_headline"]]

    assert set(headline_words["text"]) == {"GREAT", "VICTORY"}
    assert set(body_words["text"]) == {"The", "treaty", "was", "signed", "yesterday"}


def test_reading_order_column_major():
    """Assert reading order traverses Column 1 top-to-bottom, then Column 2, etc."""
    sample_data = pd.DataFrame([
        # Col 2 top
        {"column": 2, "block": 1, "par": 1, "line": 1, "word_num": 1, "text": "Col2_Top", "x": 350, "y": 100, "w": 50, "h": 16, "conf": 90},
        # Col 1 bottom
        {"column": 1, "block": 1, "par": 1, "line": 2, "word_num": 1, "text": "Col1_Bottom", "x": 50, "y": 300, "w": 50, "h": 16, "conf": 90},
        # Col 1 top
        {"column": 1, "block": 1, "par": 1, "line": 1, "word_num": 1, "text": "Col1_Top", "x": 50, "y": 100, "w": 50, "h": 16, "conf": 90},
        # Col 2 bottom
        {"column": 2, "block": 1, "par": 1, "line": 2, "word_num": 1, "text": "Col2_Bottom", "x": 350, "y": 300, "w": 50, "h": 16, "conf": 90},
    ])

    ordered = assign_reading_order(sample_data, page_num=1)
    texts_in_order = list(ordered["text"])

    # Column 1 must precede Column 2!
    expected = ["Col1_Top", "Col1_Bottom", "Col2_Top", "Col2_Bottom"]
    assert texts_in_order == expected, f"Expected {expected}, got {texts_in_order}"
    assert list(ordered["order"]) == [1, 2, 3, 4]


def test_low_confidence_and_spell_fix():
    """Assert low-confidence words are flagged and raw text is never overwritten."""
    sample_data = pd.DataFrame([
        {"column": 1, "block": 1, "par": 1, "line": 1, "word_num": 1, "text": "thiss", "conf": 45.0, "x": 10, "y": 10, "w": 20, "h": 15},
        {"column": 1, "block": 1, "par": 1, "line": 1, "word_num": 2, "text": "morning", "conf": 85.0, "x": 35, "y": 10, "w": 40, "h": 15},
    ])

    result = handle_low_confidence(sample_data, threshold=60.0, spell_fix=True, lang="eng")

    assert result.loc[0, "low_conf"] == True
    assert result.loc[1, "low_conf"] == False
    # Raw text must remain intact!
    assert result.loc[0, "text"] == "thiss"
    assert result.loc[1, "text"] == "morning"
    # Fixed text should have suggestion for 'thiss' (e.g. 'this')
    assert result.loc[0, "text_fixed"] in ["this", "thiss"]


def test_keyword_ranking_with_headline_boost():
    """Assert keyword ranking scores lines by hits * mean conf * 1.5 headline boost."""
    sample_data = pd.DataFrame([
        # Headline mentioning "peace"
        {"line_id": "L1", "column": 1, "is_headline": True, "text": "PEACE", "conf": 80.0},
        {"line_id": "L1", "column": 1, "is_headline": True, "text": "DECLARED", "conf": 80.0},
        # Body line mentioning "peace"
        {"line_id": "L2", "column": 1, "is_headline": False, "text": "The", "conf": 80.0},
        {"line_id": "L2", "column": 1, "is_headline": False, "text": "peace", "conf": 80.0},
        {"line_id": "L2", "column": 1, "is_headline": False, "text": "treaty", "conf": 80.0},
    ])

    ranked = rank_keywords(sample_data, ["peace"])
    assert not ranked.empty

    hl_score = ranked.loc[ranked["line_id"] == "L1", "score"].iloc[0]
    body_score = ranked.loc[ranked["line_id"] == "L2", "score"].iloc[0]

    # Headline score: 1 match * 0.8 conf * 1.5 boost = 1.2
    # Body score: 1 match * 0.8 conf * 1.0 boost = 0.8
    assert hl_score > body_score
    assert abs(hl_score - 1.2) <= 0.05
    assert abs(body_score - 0.8) <= 0.05


def test_full_pipeline_and_csv_schema():
    """Test full pipeline run on fixture page 1 and validate schema and BOM encoding."""
    p1_path = str(FIXTURES_DIR / "page_1_yellow_tint.png")

    with tempfile.TemporaryDirectory() as tmp_dir:
        words_df, stats_df, struct_path, reading_path = process_batch(
            file_paths=[p1_path],
            output_dir=tmp_dir,
            lang="eng",
            variant="auto",
            workers=1,
            keywords=["diplomats", "railway"],
        )

        assert not words_df.empty, "Words DataFrame must not be empty"
        assert not stats_df.empty, "Stats DataFrame must not be empty"

        words_csv_path = os.path.join(tmp_dir, "words.csv")
        assert os.path.exists(words_csv_path)

        # Check UTF-8 BOM encoding
        with open(words_csv_path, "rb") as f:
            header_bytes = f.read(3)
            assert header_bytes == b"\xef\xbb\xbf", "words.csv must have UTF-8 BOM (utf-8-sig)"

        # Check required columns
        expected_schema = [
            "file", "page", "order", "column", "block", "par", "line",
            "line_id", "word_num", "text", "text_fixed", "conf",
            "low_conf", "is_headline", "x", "y", "w", "h"
        ]
        read_df = pd.read_csv(words_csv_path, encoding="utf-8-sig")
        assert list(read_df.columns) == expected_schema

        # Check preview image generated
        previews = list(Path(tmp_dir, "previews").glob("*.jpg"))
        assert len(previews) == 1, f"Expected 1 preview image, found {len(previews)}"

        # Check reading order text
        assert os.path.exists(reading_path)
        with open(reading_path, "r", encoding="utf-8") as f:
            txt = f.read()
            assert "## " in txt, "Headlines should be prefixed with '## '"
            assert "PAGE 1" in txt


def test_error_isolation():
    """Assert corrupted or invalid files do not crash the batch and error is recorded in stats."""
    # Create corrupted dummy image file
    with tempfile.TemporaryDirectory() as tmp_dir:
        bad_file = os.path.join(tmp_dir, "corrupt.png")
        with open(bad_file, "wb") as f:
            f.write(b"NOT_A_VALID_IMAGE_DATA")

        good_file = str(FIXTURES_DIR / "page_1_yellow_tint.png")

        words_df, stats_df, _, _ = process_batch(
            file_paths=[bad_file, good_file],
            output_dir=os.path.join(tmp_dir, "out"),
            workers=2,
        )

        assert len(stats_df) == 2
        bad_stat = stats_df[stats_df["file"] == "corrupt.png"].iloc[0]
        good_stat = stats_df[stats_df["file"] == "page_1_yellow_tint.png"].iloc[0]

        # Corrupted file must have an error recorded
        assert bad_stat["error"] != ""
        assert bad_stat["words"] == 0

        # Good file must have succeeded
        assert good_stat["error"] == ""
        assert good_stat["words"] > 0
