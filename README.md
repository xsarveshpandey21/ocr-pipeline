# Industrial CPU-Only OCR Pipeline for Degraded Newspapers

A high-throughput, **CPU-only** OCR pipeline and interactive Gradio application engineered specifically for historical, fragile, and severely degraded multi-column newspaper scans (yellowed/stained newsprint, uneven illumination gradients, scan shadows, ink bleed-through, skew, tiny dense typography, and vertical/horizontal ruling lines).

---

## 🌟 Key Features

- **100% CPU Execution**: Zero GPU/CUDA dependencies. Runs smoothly on multi-core consumer laptops and servers.
- **Multiprocessing Spawn Engine**: Uses `multiprocessing.get_context("spawn")` with `OMP_THREAD_LIMIT=1` to prevent CPU oversubscription and maximize parallel throughput.
- **Memory-Safe Streaming**: Renders and processes pages strictly on-demand inside isolated worker processes. Never loads all 90+ high-res images into RAM.
- **Horizontal Projection Deskew**: Automatically searches and corrects document skew in the $[-4.0^\circ, +4.0^\circ]$ range.
- **Illumination Flattening**: Large-kernel median division eliminates yellowed newsprint, uneven lighting, and gutter shadows.
- **Morphological Ruling Line Erasure**: Detects and paints column separators and horizontal dividing rules white before character recognition.
- **Adaptive Variant Selector (`auto`)**: Tests `raw`, `light`, and `heavy` cleanup variants on a middle sample band and selects the highest-scoring variant before full-page OCR.
- **Column-Major Reading Order**: Vertical projection-profile gutter detection segments columns left-to-right, sorting lines top-to-bottom ($C_1 \to C_2 \to \dots \to C_k$).
- **Headline Detection**: Automatically identifies headlines when median word height exceeds $1.6\times$ page body-line height (`is_headline = True`).
- **Non-Destructive Spell Correction**: Flags low-confidence words ($<60\%$) and provides English suggestions in `text_fixed` without mutating raw OCR text.
- **Keyword Relevance Ranking**: Relevancy scoring across lines with a $1.5\times$ boost for headlines.
- **Cross-Platform**: Tested on Windows, Linux, and macOS with native environment variable discovery (`TESSERACT_CMD`, `POPPLER_PATH`).

---

## 📋 System Prerequisites & Installation

### 1. Python Environment
Requires **Python 3.10+**. Clone repository and install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Tesseract 5 & Poppler Setup

#### **Windows**
Using Scoop (recommended user-space install):
```powershell
scoop install tesseract poppler
```
Or via WinGet / UB-Mannheim installer:
```powershell
winget install --id UB-Mannheim.TesseractOCR -e
```
*Optional Language Packs (e.g. Hindi `hin`, Urdu `urd`):*
Download `.traineddata` files from [tessdata_fast](https://github.com/tesseract-ocr/tessdata_fast) and place them into your `tessdata` directory (e.g. `C:\Users\<username>\scoop\apps\tesseract\current\tessdata`).

#### **Linux (Ubuntu / Debian)**
```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr poppler-utils tesseract-ocr-hin tesseract-ocr-urd
```

#### **macOS (Homebrew)**
```bash
brew install tesseract poppler tesseract-lang
```

#### **Custom Path Overrides**
You can configure explicit binary paths using environment variables:
```bash
export TESSERACT_CMD="/path/to/tesseract"
export POPPLER_PATH="/path/to/poppler/bin"
```

---

## 🚀 Quickstart & Usage

### 1. Launch Gradio Web Interface
```bash
python app.py
```
Open your browser at `http://localhost:7860`.
- **Upload Scans**: Multi-file drag & drop (PDF, PNG, JPG, TIFF).
- **Configure**: Select installed languages (e.g., `eng`, `eng+hin`), cleanup variant (`auto`), and optional keywords.
- **Track Progress**: Live progress indicator updating per finished page.
- **Inspect**: Interactive confidence-map visual previews, reading-order text reader, sortable words table, keyword relevance rankings, and one-click ZIP package download.

### 2. Headless CLI Processing
Run batch OCR from the command line:
```bash
# Process a folder of scans using 4 parallel workers
python ocr_pipeline.py samples/ --lang eng --workers 4 --output-dir output/

# Process with keyword relevance ranking and spell correction
python ocr_pipeline.py newspaper.pdf --keywords treaty parliament harvest --spell-fix --workers 4
```

---

## 📊 Output Deliverables & Schema

All outputs are exported to the target `--output-dir` (default `output/`):

### 1. `words.csv`
Encoded in **UTF-8 with BOM (`utf-8-sig`)** for Excel compatibility with English, Hindi, and Urdu characters. Contains one row per recognized word:

| Field | Type | Description |
|---|---|---|
| `file` | string | Source document filename |
| `page` | int | 1-indexed page number |
| `order` | int | Column-major reading sequence (1 to $N$) |
| `column` | int | 1-indexed column index |
| `block` | int | Tesseract block ID |
| `par` | int | Tesseract paragraph ID |
| `line` | int | Tesseract line ID |
| `line_id` | string | Global identifier, e.g. `P001_C01_L0004` |
| `word_num` | int | Word index within line |
| `text` | string | **Raw OCR word text (strictly immutable)** |
| `text_fixed` | string | Spell-corrected word (if enabled & low confidence) |
| `conf` | float | OCR confidence score (0 to 100) |
| `low_conf` | bool | `True` if `conf < low_conf_thresh` (default 60) |
| `is_headline` | bool | `True` if word line height $> 1.6\times$ page body median |
| `x, y, w, h` | int | Bounding box coordinates in full-page pixel space |

### 2. `structure.json`
Hierarchical nested representation: `pages -> columns -> lines -> words`:
```json
{
  "page_1": {
    "column_1": {
      "P001_C01_L0001": {
        "is_headline": true,
        "words": [
          {
            "order": 1,
            "word_num": 1,
            "text": "HISTORIC",
            "text_fixed": "HISTORIC",
            "conf": 96.0,
            "low_conf": false,
            "box": [80, 190, 185, 36]
          }
        ]
      }
    }
  }
}
```

### 3. `reading_order.txt`
Human-readable transcribed article stream:
- Column headers: `--- COLUMN 1 ---`
- Headlines prefixed with `## `
- Low-confidence words wrapped in angle brackets `<word>`

### 4. `page_stats.csv`
Execution telemetry per page: `file, page, words, mean_conf, low_conf_pct, num_columns, chosen_variant, skew_angle, seconds, error`.

### 5. `keyword_hits.csv`
When keywords are specified, ranks occurrences by:
$$\text{Score} = (\text{Keyword Matches}) \times \left(\frac{\text{Mean Conf}}{100}\right) \times (1.5 \text{ if is\_headline else } 1.0)$$

### 6. `previews/page_XXX_annotated.jpg`
Direct-to-disk annotated visual inspection previews:
- **Green boxes**: High confidence ($\ge 70\%$)
- **Orange boxes**: Moderate confidence ($50-69\%$)
- **Red boxes**: Low confidence ($< 50\%$)
- **Blue bold boxes**: Classified headlines
- **Magenta vertical lines**: Detected column gutters

---

## 💡 Tuning Tips for Severe Degradation

| Symptom | Recommended Configuration Adjustment |
|---|---|
| **Heavily yellowed / stained paper** | Default background flattening handles this automatically. If staining is intense, set `--variant light` or `--variant auto`. |
| **Severe paper grain & bleed-through** | Use `--variant heavy` (fast bilateral smoothing + adaptive thresholding) to eliminate background speckles. |
| **Very tiny / dense 6-point typography** | Increase rendering DPI: `--dpi 350` or `--dpi 400`. |
| **Non-standard column layouts** | Set `--psm 4` (single column mode) with `--do-column-split` enabled. If layout is chaotic, disable column splitting (`--no-columns`) to let Tesseract use full-page layout analysis (`--psm 3`). |
| **Multi-lingual newspapers (e.g. English + Hindi)** | Pass combined language flags: `--lang eng+hin`. |

---

## ⚠️ Known Limitations & Future Improvements

1. **Multi-Column Spanning Headlines**: Banner headlines spanning across all columns at the top of a page are segmented at the column boundaries. Reading order places the banner's left segment in Column 1 and the right segment in Column 2. *Future improvement: Connected-component headline merging across gutters prior to column clipping.*
2. **Severe 3D Page Curvature**: Historical bound newspapers often exhibit cylindrical distortion near the spine. The horizontal projection deskewer corrects planar affine rotation but does not dewarp 3D cylindrical book curves. *Future improvement: Thin-plate spline (TPS) mesh unwarping.*
3. **Optional CPU Second-Pass**: Very low-confidence words can be re-evaluated via an optional lightweight CPU PaddleOCR model flag if installed.

---

## 🧪 Automated Test Suite

Run unit and integration tests covering deskew, background flattening, line removal, column detection, headline tagging, and CSV schema:
```bash
python -m pytest -v tests/test_pipeline.py
```
All 10 tests execute and pass in $\approx 8.5$ seconds.
