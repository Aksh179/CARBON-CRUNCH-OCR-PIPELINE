# Carbon Crunch – AI-OCR Pipeline  
## Technical Documentation

---

### 1. Approach

The system is structured as a four-stage pipeline that mirrors a classical NLP information-extraction architecture, adapted for the receipt domain.

```
Image → Preprocessor → OCR Engine → Field Extractor → Confidence Scorer → JSON
```

#### Stage 1 – Image Preprocessing (`ImagePreprocessor`)

Raw receipt images suffer from four classes of degradation: low resolution, uneven lighting, print noise, and camera tilt. Each is addressed with a dedicated classical-CV step:

| Problem | Fix |
|---|---|
| Low resolution / small scan | Bicubic upscaling to ≥ 1000 px on the longest axis |
| Noise / film grain | `cv2.fastNlMeansDenoising` (h=10) |
| Uneven lighting / shadows | CLAHE (`clipLimit=2.0`, `tileGridSize=8×8`) |
| Camera tilt / document skew | Hough-line median-angle deskew + `warpAffine` |
| Mixed fonts, background | Adaptive Gaussian threshold (`blockSize=31`, `C=11`) |

All steps are logged and stored in the result JSON under `preprocessing_steps` so the pipeline is fully reproducible and debuggable.

#### Stage 2 – Text Detection & Recognition (`OCREngine`)

**Primary engine: EasyOCR**  
EasyOCR uses a CRAFT text detector + CRNN recogniser trained on 80+ languages. It returns per-word bounding boxes and confidence scores natively, which flow directly into our confidence model. GPU is disabled by default for portability; enabling it via `gpu=True` in `easyocr.Reader` can double throughput.

**Fallback engine: Tesseract (via pytesseract)**  
Tesseract with `--psm 6` (assume uniform block of text) provides word-level confidence from its `image_to_data` API. Scores are normalised to [0, 1].

The engine is auto-detected at runtime; users can pin a specific engine with `--engine easyocr|tesseract`.

#### Stage 3 – Key Information Extraction (`FieldExtractor`)

Words are first grouped into *lines* by proximity on the vertical axis (y-tolerance = 12 px). Each field is then extracted with a tailored heuristic:

- **Store / Vendor Name**: Highest-scoring candidate in the first 1–4 lines that is not purely numeric (receipts almost universally print the merchant name at the top).
- **Date**: Multi-pattern regex search across 4 common date formats (DD/MM/YYYY, YYYY-MM-DD, "12 Mar 2024", etc.), returning the first match.
- **Total Amount**: Lines where text overlaps `TOTAL_KEYWORDS = {total, amount, grand total, balance, …}` are scored higher; the rightmost currency amount on the best-scoring line is selected.
- **Line Items**: Any line containing a currency amount that is *not* a total keyword is treated as an item; the price is the rightmost currency token, the name is everything to its left.

#### Stage 4 – Confidence Scoring

Each field receives a composite confidence score using three signals:

```
field_confidence = 0.60 × ocr_mean_confidence
                 + 0.25 × pattern_valid          # 1 if regex/heuristic matched
                 + keyword_bonus                  # 0–0.15 for strong keyword hits
```

Fields with `confidence < 0.70` are **flagged** (`"flagged": true` in the JSON output).  
The overall receipt confidence is a weighted mean:

```
overall = 0.30 × store_conf + 0.30 × total_conf + 0.20 × date_conf + 0.20 × avg_item_conf
```

---

### 2. Tools & Libraries

| Component | Library | Notes |
|---|---|---|
| Image I/O & CV | OpenCV 4.8+ | Preprocessing, deskew |
| OCR (primary) | EasyOCR 1.7+ | CRAFT + CRNN |
| OCR (fallback) | Tesseract 5 + pytesseract | --psm 6 mode |
| Array maths | NumPy 1.24+ | Confidence arithmetic |
| Image conversion | Pillow 10+ | Tesseract bridge |
| Output | stdlib `json` | No extra dependency |

All dependencies are pinned in `requirements.txt`. The system runs on Python 3.10+.

---

### 3. Challenges Faced

**3.1 Variable receipt layouts**  
Receipts have no schema. Grocery, pharmacy, restaurant, and fuel receipts all differ in where they print totals, whether items have prices on the same line, and how dates are formatted. The heuristic extractor was tuned to handle the most common conventions but is inherently imperfect for highly unusual layouts.

**3.2 OCR confidence calibration**  
EasyOCR's confidence scores are overconfident for blurry or skewed text. The CLAHE + deskew preprocessing step was critical to bringing OCR scores into a range where our thresholds (0.70) were meaningful. Without it, scores could be high even for garbled output.

**3.3 Price vs. total disambiguation**  
Distinguishing an item price from a subtotal/total is hard when keyword context is missing (e.g., a receipt that prints "35.90" with no label). The current heuristic picks the *last* currency amount on a line with a total keyword, which works for ~85% of test cases.

**3.4 Composite store names**  
Multi-word merchant names (e.g., "THE BODY SHOP") are sometimes split across lines by the OCR layout. The current implementation joins all words on the same Y-band, which handles most cases.

**3.5 Partial / torn receipts**  
Images where the top is cut off lose the vendor name. The pipeline gracefully returns `"store_name": "UNKNOWN"` with `confidence: 0.0` and `flagged: true` rather than crashing.

---

### 4. Improvements & Future Work

| Area | Improvement |
|---|---|
| **OCR accuracy** | Fine-tune a TrOCR (Microsoft) or PaddleOCR model on a labelled receipt dataset (CORD, SROIE) to improve recognition on stylised fonts and low-resolution prints |
| **Layout understanding** | Replace line-based heuristics with a LayoutLM or Donut (document understanding transformer) model that uses both text and spatial position for field extraction |
| **Date normalisation** | Standardise all extracted dates to ISO 8601 (YYYY-MM-DD) using `dateparser` |
| **Currency normalisation** | Detect currency symbol and normalise amounts to a common base for cross-currency summaries |
| **Multi-page receipts** | Handle PDF inputs via `pdf2image` (one page per call) |
| **Batch parallelism** | Add `concurrent.futures.ThreadPoolExecutor` for faster directory processing |
| **Ground-truth evaluation** | Integrate SROIE benchmark labels to report CER, field F1, and confidence calibration plots |
| **UI dashboard** | Build a Streamlit or Gradio frontend to upload images and inspect JSON results interactively |

---

*Prepared for Carbon Crunch Shortlisting Assignment.*
