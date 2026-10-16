#!/usr/bin/env python3
"""
Generate 3 realistic degraded synthetic newspaper pages for testing and benchmarking.
Simulates:
- Multi-column newspaper layout (3 columns with headlines and vertical ruling lines)
- Aged paper discoloration (yellowed/sepia tint)
- Uneven scanner lighting / vignetting / shadows
- Show-through / ink bleed from verso page
- Skew (-4 to +4 degrees)
- Ink degradation, grain, and salt-and-pepper noise
"""

import math
import os
import random
from pathlib import Path
from typing import List
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def get_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    """Return Times New Roman or Arial font if available, else default."""
    font_paths = [
        r"C:\Windows\Fonts\timesbd.ttf" if bold else r"C:\Windows\Fonts\times.ttf",
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "/System/Library/Fonts/Times.ttc",
    ]
    for p in font_paths:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


def wrap_text(text: str, font: ImageFont.ImageFont, max_width: int) -> List[str]:
    """Break long text into wrapped lines that strictly fit within max_width."""
    words = text.split()
    lines: List[str] = []
    curr: List[str] = []
    for w in words:
        test = " ".join(curr + [w])
        bbox = font.getbbox(test)
        if (bbox[2] - bbox[0]) <= max_width:
            curr.append(w)
        else:
            if curr:
                lines.append(" ".join(curr))
            curr = [w]
    if curr:
        lines.append(" ".join(curr))
    return lines


def render_base_newspaper(width: int = 1600, height: int = 2200) -> Image.Image:
    """Render clean multi-column newspaper layout with headers, headlines, columns, and rules."""
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    banner_font = get_font(42, bold=True)
    sub_banner_font = get_font(20, bold=False)
    headline_font = get_font(36, bold=True)
    body_font = get_font(18, bold=False)

    margin_x = 80
    margin_top = 70

    # 1. Newspaper Title Banner
    banner_text = "THE DAILY CHRONICLE"
    draw.text((margin_x, margin_top), banner_text, fill=(10, 10, 10), font=banner_font)
    date_text = "Vol. LXXVIII No. 24,115  *  Special Edition  *  London, October 16, 1924  *  Price Two Pence"
    draw.text((margin_x, margin_top + 55), date_text, fill=(50, 50, 50), font=sub_banner_font)

    # Double banner ruling lines
    draw.line([(margin_x, margin_top + 90), (width - margin_x, margin_top + 90)], fill=(0, 0, 0), width=3)
    draw.line([(margin_x, margin_top + 96), (width - margin_x, margin_top + 96)], fill=(0, 0, 0), width=1)

    # 2. Main Page Headline
    hl_text = "HISTORIC ACCORD SIGNED AFTER EXTENSIVE DEBATE"
    draw.text((margin_x, margin_top + 120), hl_text, fill=(0, 0, 0), font=headline_font)

    # Subline below headline
    sub_hl = "Delegates Reach Unanimous Consensus on Regional Economic Cooperation and Commerce"
    draw.text((margin_x, margin_top + 175), sub_hl, fill=(30, 30, 30), font=get_font(22, bold=True))

    draw.line([(margin_x, margin_top + 215), (width - margin_x, margin_top + 215)], fill=(0, 0, 0), width=2)

    # 3. Three Column Setup
    content_top = margin_top + 235
    content_bottom = height - 100
    total_col_width = width - 2 * margin_x
    gutter_width = 40
    num_cols = 3
    col_width = (total_col_width - (num_cols - 1) * gutter_width) // num_cols

    columns_x = [margin_x + i * (col_width + gutter_width) for i in range(num_cols)]

    # Draw vertical ruling lines in the middle of gutters
    for i in range(num_cols - 1):
        rule_x = columns_x[i] + col_width + gutter_width // 2
        draw.line([(rule_x, content_top), (rule_x, content_bottom)], fill=(40, 40, 40), width=2)

    # Column 1 Articles
    col1_articles = [
        ("GENEVA DIPLOMATS CONCLUDE PACT", [
            "Yesterday afternoon the plenipotentiaries assembled at the great hall and formally appended their signatures to the solemn declaration.",
            "The discussions had continued without intermission for over thirty hours, with delegates from every sovereign state presenting their terms.",
            "Public enthusiasm was manifest throughout the city as bells rang out across the harbor to announce the joyful news.",
            "Premier Baldwin addressed the cheering throngs from the balcony, affirming that peace and prosperity shall now govern our future.",
            "Trade agreements are scheduled to take effect from the first of the month, reopening vital maritime shipping lanes and commercial markets.",
        ]),
        ("RAILWAY EXTENSIONS APPROVED", [
            "The Minister of Transport announced today that Parliament has sanctioned the proposed northern railway trunk line connecting all industrial centers.",
            "Construction will commence immediately, providing employment for thousands of skilled artisans and mechanics over the coming winter months.",
            "New steel locomotives of unprecedented power have been commissioned from the northern foundries to handle the burgeoning freight volume.",
        ]),
    ]

    # Column 2 Articles
    col2_articles = [
        ("FINANCIAL MARKETS RALLY SHARPLY", [
            "Securities on the Royal Stock Exchange registered dramatic advances immediately upon receipt of the diplomatic dispatch from the continent.",
            "Consols appreciated by two full points before the noon closing bell, with shipping and textile shares leading the spirited advance.",
            "Foreign exchange rates displayed commendable stability throughout the session, with the pound sterling maintaining firm resistance against all rivals.",
            "Prominent bankers expressed confident expectations of lower discount rates.",
        ]),
        ("AGRICULTURAL YIELDS SURPASS ESTIMATES", [
            "Reports from the harvest districts indicate that despite unseasonable rains, the grain harvest has yielded results far in advance of expectations.",
            "Wheat granaries in the eastern counties report record bushel collections, assuring adequate winter bread provisions and stabilizing commodity tariffs.",
            "Dairy farmers likewise report sustained milk yields throughout the season.",
        ]),
    ]

    # Column 3 Articles
    col3_articles = [
        ("SCIENTIFIC EXPEDITION RETURNS HOME", [
            "The intrepid voyagers of the Arctic Research vessel have safely docked, bringing home extensive collections of geological specimens and charts.",
            "Dr. Henderson reported that extensive uncharted straits were surveyed under grueling conditions of winter ice and persistent polar gales.",
            "A public lecture illustrating their dramatic discoveries with stereopticon slides will be delivered before the Royal Geographical Society on Friday.",
            "Admiralty officials have commended the crew for unexampled perseverance.",
        ]),
        ("WEATHER OUTLOOK FOR THE REGION", [
            "Moderate to fresh southwesterly breezes with intermittent sunshine across all districts.",
            "Slight risk of coastal squalls toward twilight, clearing before dawn.",
            "Barometer: 30.15 inches and rising steadily across all maritime sectors.",
        ]),
    ]

    articles_per_col = [col1_articles, col2_articles, col3_articles]

    for col_idx, articles in enumerate(articles_per_col):
        cx = columns_x[col_idx]
        curr_y = content_top

        for art_title, raw_paragraphs in articles:
            # Sub-headline inside column
            draw.text((cx, curr_y), art_title, fill=(0, 0, 0), font=get_font(20, bold=True))
            curr_y += 28
            draw.line([(cx, curr_y), (cx + col_width, curr_y)], fill=(80, 80, 80), width=1)
            curr_y += 10

            for p_text in raw_paragraphs:
                wrapped_lines = wrap_text(p_text, body_font, col_width - 10)
                for line_str in wrapped_lines:
                    draw.text((cx, curr_y), line_str, fill=(15, 15, 15), font=body_font)
                    curr_y += 24
                curr_y += 8  # paragraph space

            curr_y += 25

    # Bottom footer line
    draw.line([(margin_x, content_bottom + 20), (width - margin_x, content_bottom + 20)], fill=(0, 0, 0), width=1)
    draw.text((margin_x, content_bottom + 30), "Printed and Published at the Fleet Street Offices of The Daily Chronicle.", fill=(80, 80, 80), font=sub_banner_font)

    return img


def apply_page_1_degradations(img: Image.Image) -> Image.Image:
    """Page 1: Yellow aged paper, mild illumination gradient, subtle bleed-through."""
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    h, w = cv_img.shape[:2]

    # 1. Aged yellow paper tint (multiply with warm sepia tone)
    tint = np.array([[[195 / 255.0, 235 / 255.0, 248 / 255.0]]], dtype=np.float32)
    tinted = (cv_img.astype(np.float32) * tint).clip(0, 255).astype(np.uint8)

    # 2. Uneven lighting gradient (brighter top-left, slightly shaded bottom-right)
    x = np.linspace(1.0, 0.85, w, dtype=np.float32)
    y = np.linspace(1.0, 0.88, h, dtype=np.float32)
    grad_x, grad_y = np.meshgrid(x, y)
    grad = (grad_x * grad_y)[:, :, np.newaxis]

    shaded = (tinted.astype(np.float32) * grad).clip(0, 255).astype(np.uint8)

    # 3. Subtle bleed-through / show-through text from verso page
    verso = cv2.flip(cv_img, 1)
    verso_faint = (255 - verso) * 0.04
    combined = np.clip(shaded.astype(np.float32) - verso_faint, 0, 255).astype(np.uint8)

    # 4. Light Gaussian noise / paper grain
    noise = np.random.normal(0, 3, (h, w, 3))
    degraded = np.clip(combined.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return Image.fromarray(cv2.cvtColor(degraded, cv2.COLOR_BGR2RGB))


def apply_page_2_degradations(img: Image.Image) -> Image.Image:
    """Page 2: Same layout with intentional +2.5 degree skew and scanner edge shadow."""
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    h, w = cv_img.shape[:2]

    # 1. Yellow tint
    tint = np.array([[[185 / 255.0, 225 / 255.0, 242 / 255.0]]], dtype=np.float32)
    tinted = (cv_img.astype(np.float32) * tint).clip(0, 255).astype(np.uint8)

    # 2. Deep scanner spine gutter shadow on the left side
    spine_shadow = np.ones((h, w), dtype=np.float32)
    spine_w = int(w * 0.15)
    spine_curve = np.linspace(0.70, 1.0, spine_w, dtype=np.float32) ** 2
    spine_shadow[:, :spine_w] = spine_curve[np.newaxis, :]
    shaded = (tinted.astype(np.float32) * spine_shadow[:, :, np.newaxis]).clip(0, 255).astype(np.uint8)

    # 3. Add +2.5 degree skew with white border
    angle = 2.5
    center = (w / 2.0, h / 2.0)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        shaded,
        rot_mat,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(240, 240, 240),
    )

    # 4. Paper grain
    noise = np.random.normal(0, 4, (h, w, 3))
    degraded = np.clip(rotated.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return Image.fromarray(cv2.cvtColor(degraded, cv2.COLOR_BGR2RGB))


def apply_page_3_degradations(img: Image.Image) -> Image.Image:
    """Page 3 (Worst Case): Heavy stains, blotches, salt-and-pepper noise, -1.5 deg skew, faded ink."""
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    h, w = cv_img.shape[:2]

    # 1. Darker aged sepia/yellow tint
    tint = np.array([[[170 / 255.0, 215 / 255.0, 235 / 255.0]]], dtype=np.float32)
    tinted = (cv_img.astype(np.float32) * tint).clip(0, 255).astype(np.uint8)

    # 2. Add realistic stain blotches (water stains / coffee stains)
    stain_mask = np.ones((h, w), dtype=np.float32)
    cv2.circle(stain_mask, (int(w * 0.85), int(h * 0.2)), int(w * 0.18), 0.78, -1)
    cv2.ellipse(stain_mask, (int(w * 0.5), int(h * 0.85)), (int(w * 0.25), int(h * 0.12)), 25, 0, 360, 0.72, -1)
    stain_mask = cv2.GaussianBlur(stain_mask, (101, 101), 30)

    stained = (tinted.astype(np.float32) * stain_mask[:, :, np.newaxis]).clip(0, 255).astype(np.uint8)

    # 3. Illumination gradient
    x = np.linspace(1.0, 0.75, w, dtype=np.float32)
    y = np.linspace(0.80, 1.0, h, dtype=np.float32)
    grad_x, grad_y = np.meshgrid(x, y)
    shaded = (stained.astype(np.float32) * (grad_x * grad_y)[:, :, np.newaxis]).clip(0, 255).astype(np.uint8)

    # 4. Salt and pepper noise
    sp_mask = np.random.rand(h, w)
    shaded[sp_mask < 0.005] = (30, 30, 30)  # Pepper
    shaded[sp_mask > 0.995] = (250, 250, 250)  # Salt

    # 5. Skew -1.5 degrees
    angle = -1.5
    center = (w / 2.0, h / 2.0)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        shaded,
        rot_mat,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(235, 235, 235),
    )

    # 6. Slight ink fade / blur
    degraded = cv2.GaussianBlur(rotated, (3, 3), 0.5)
    return Image.fromarray(cv2.cvtColor(degraded, cv2.COLOR_BGR2RGB))


def generate_all_fixtures(output_dir: str = "tests/fixtures"):
    """Generate all 3 degraded synthetic newspaper pages and save to output_dir."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("Generating base newspaper layout with wrapped column text...")
    base_img = render_base_newspaper(width=1600, height=2200)

    print("Synthesizing Page 1 (yellow tint, mild gradient, bleed-through)...")
    p1 = apply_page_1_degradations(base_img)
    p1_path = out_path / "page_1_yellow_tint.png"
    p1.save(p1_path)
    print(f"Saved: {p1_path}")

    print("Synthesizing Page 2 (+2.5 deg skew, scanner spine shadow)...")
    p2 = apply_page_2_degradations(base_img)
    p2_path = out_path / "page_2_skewed.png"
    p2.save(p2_path)
    print(f"Saved: {p2_path}")

    print("Synthesizing Page 3 (severe degradation, stains, noise, -1.5 deg skew)...")
    p3 = apply_page_3_degradations(base_img)
    p3_path = out_path / "page_3_worst_case.png"
    p3.save(p3_path)
    print(f"Saved: {p3_path}")

    print("All 3 synthetic fixtures successfully created in:", output_dir)


if __name__ == "__main__":
    generate_all_fixtures()
