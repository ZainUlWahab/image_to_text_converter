"""
Layout Analysis Module
Detects text line positions and infers bold/italic/alignment from visual features.
This complements OCR text extraction with formatting information.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Literal, Tuple

import numpy as np
import cv2
from PIL import Image, ImageDraw


Alignment = Literal["left", "center", "right"]


@dataclass(frozen=True)
class LineStyle:
    """Formatting style for a text line."""
    bold: bool
    italic: bool
    alignment: Alignment


@dataclass(frozen=True)
class LineBox:
    """Bounding box for a text line with style information."""
    x: int
    y: int
    w: int
    h: int
    style: LineStyle


@dataclass(frozen=True)
class LayoutAnalysis:
    """Complete layout analysis result."""
    width: int
    height: int
    lines: List[LineBox]


def _pil_to_bgr(image: Image.Image) -> np.ndarray:
    """Convert PIL Image to OpenCV BGR format."""
    rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _binarize(gray: np.ndarray) -> np.ndarray:
    """
    Robust binarization for scanned documents (white background).
    Returns binary image with ink pixels as 255 (white on black).
    """
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thr = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,  # Invert: text becomes white
        31,
        15,
    )
    return thr


def _find_line_boxes(binary_inv: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Find text line bounding boxes by merging characters horizontally.
    
    Args:
        binary_inv: Binary image with ink pixels as 255
        
    Returns:
        List of (x, y, width, height) tuples for each line
    """
    h, w = binary_inv.shape[:2]
    
    # Horizontal kernel to merge characters into lines
    kernel_w = max(20, w // 40)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 3))
    merged = cv2.dilate(binary_inv, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes: List[Tuple[int, int, int, int]] = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        
        # Filter obvious noise (too small)
        if bw < w * 0.08 or bh < 8:
            continue
        
        boxes.append((x, y, bw, bh))
    
    # Sort by vertical position, then horizontal
    boxes.sort(key=lambda b: (b[1], b[0]))
    
    # Merge very close boxes on the same line
    merged_boxes: List[Tuple[int, int, int, int]] = []
    for x, y, bw, bh in boxes:
        if not merged_boxes:
            merged_boxes.append((x, y, bw, bh))
            continue
        
        px, py, pw, ph = merged_boxes[-1]
        
        # Check if on same row (vertical centers are close)
        same_row = abs((y + bh // 2) - (py + ph // 2)) < max(10, int(0.35 * max(bh, ph)))
        close = (x - (px + pw)) < 20
        
        if same_row and close:
            # Merge boxes
            nx = min(px, x)
            ny = min(py, y)
            nx2 = max(px + pw, x + bw)
            ny2 = max(py + ph, y + bh)
            merged_boxes[-1] = (nx, ny, nx2 - nx, ny2 - ny)
        else:
            merged_boxes.append((x, y, bw, bh))
    
    return merged_boxes


def _infer_alignment(box: Tuple[int, int, int, int], page_w: int) -> Alignment:
    """
    Infer text alignment based on horizontal position.
    
    Args:
        box: (x, y, width, height) tuple
        page_w: Page width
        
    Returns:
        Alignment: 'left', 'center', or 'right'
    """
    x, _, bw, _ = box
    left_margin = x / max(1, page_w)
    right_margin = (page_w - (x + bw)) / max(1, page_w)
    
    # Center: both margins similar and not too close to edges
    if abs(left_margin - right_margin) < 0.08 and left_margin > 0.10 and right_margin > 0.10:
        return "center"
    
    # Right: right margin tiny but left margin large
    if right_margin < 0.06 and left_margin > 0.18:
        return "right"
    
    return "left"


def _ink_density(binary_inv_crop: np.ndarray) -> float:
    """
    Calculate ink density (ratio of ink pixels to total pixels).
    
    Args:
        binary_inv_crop: Binary image crop with ink as 255
        
    Returns:
        Ink density ratio (0-1)
    """
    return float(np.mean(binary_inv_crop > 0))


def _infer_bold(binary_inv_crop: np.ndarray) -> bool:
    """
    Infer if text is bold based on ink density.
    Bold text has thicker strokes = higher ink density.
    
    Args:
        binary_inv_crop: Binary image crop of the line
        
    Returns:
        True if likely bold
    """
    density = _ink_density(binary_inv_crop)
    return density > 0.18


def _infer_italic(gray_crop: np.ndarray) -> bool:
    """
    Infer if text is italic based on stroke slant angle.
    Italic text typically has 10-25° slant.
    
    Args:
        gray_crop: Grayscale image crop of the line
        
    Returns:
        True if likely italic
    """
    # Detect edges
    edges = cv2.Canny(gray_crop, 50, 150, apertureSize=3)
    
    # Find lines using Hough transform
    lines = cv2.HoughLinesP(
        edges, 
        1, 
        np.pi / 180, 
        threshold=60, 
        minLineLength=20, 
        maxLineGap=8
    )
    
    if lines is None:
        return False
    
    # Analyze angles
    angles = []
    for (x1, y1, x2, y2) in lines[:, 0]:
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0:
            continue
        
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Keep near-horizontal strokes
        if -40 <= angle <= 40:
            angles.append(angle)
    
    if len(angles) < 6:
        return False
    
    # Italic if median angle is 10-25°
    median = float(np.median(angles))
    return 10.0 <= abs(median) <= 25.0


def analyze_layout(image: Image.Image) -> LayoutAnalysis:
    """
    Analyze document layout to detect text lines and formatting.
    
    Args:
        image: PIL Image of the document
        
    Returns:
        LayoutAnalysis with detected line boxes and styles
    """
    # Convert to OpenCV format
    bgr = _pil_to_bgr(image)
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    
    # Binarize
    bin_inv = _binarize(gray)
    
    # Find line bounding boxes
    boxes = _find_line_boxes(bin_inv)
    
    # Analyze each line
    lines: List[LineBox] = []
    for (x, y, bw, bh) in boxes:
        # Crop line region
        crop_bin = bin_inv[max(0, y):min(h, y + bh), max(0, x):min(w, x + bw)]
        crop_gray = gray[max(0, y):min(h, y + bh), max(0, x):min(w, x + bw)]
        
        # Infer formatting
        alignment = _infer_alignment((x, y, bw, bh), page_w=w)
        bold = _infer_bold(crop_bin)
        italic = _infer_italic(crop_gray)
        
        lines.append(
            LineBox(
                x=int(x),
                y=int(y),
                w=int(bw),
                h=int(bh),
                style=LineStyle(bold=bold, italic=italic, alignment=alignment),
            )
        )
    
    return LayoutAnalysis(width=int(w), height=int(h), lines=lines)


def render_layout_debug(image: Image.Image, layout: LayoutAnalysis) -> Image.Image:
    """
    Render debug visualization of detected layout.
    
    Args:
        image: Original PIL Image
        layout: Layout analysis result
        
    Returns:
        PIL Image with layout boxes and labels overlaid
    """
    out = image.copy().convert("RGB")
    draw = ImageDraw.Draw(out)
    
    for line in layout.lines:
        # Draw bounding box
        x1, y1 = line.x, line.y
        x2, y2 = line.x + line.w, line.y + line.h
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
        
        # Draw label with style information
        label = f"{line.style.alignment} | {'B' if line.style.bold else '-'}{'I' if line.style.italic else '-'}"
        draw.text((x1, max(0, y1 - 14)), label, fill=(255, 0, 0))
    
    return out
