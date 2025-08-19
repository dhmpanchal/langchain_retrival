import fitz
import tempfile
from pdf2image import convert_from_path
import boto3
import re
from math import ceil
import constant
from io import BytesIO
import random

# Reuse AWS Textract client from your settings if available
textract_client = boto3.client('textract',
                                aws_access_key_id= constant.aws_access_key_id,
                                aws_secret_access_key=constant.aws_secret_access_key,
                                region_name=constant.region_name)

def find_term_coordinates_with_ocr_v1(pdf_path, diagnoses, allow_partial=True):
    """
    Performs OCR on the PDF using AWS Textract and finds term coordinates.
    Works even if PDF has no text layer (scanned PDF).
    """
    results = {}

    # Convert PDF pages to images for Textract
    target_dpi = 144
    pdf_pages = convert_from_path(pdf_path, dpi=target_dpi)
    textract_dpi = 72
    scaling_factor = target_dpi / textract_dpi

    all_words_and_boxes = []

    for page_index, page in enumerate(pdf_pages):
        width, height = page.size
        byte_arr = BytesIO()
        page.save(byte_arr, format='PNG')
        image_bytes = byte_arr.getvalue()

        response = textract_client.analyze_document(
            Document={'Bytes': image_bytes},
            FeatureTypes=['LAYOUT']
        )

        words_and_boxes = []
        for block in response['Blocks']:
            if block['BlockType'] == "WORD":
                box = block['Geometry']['BoundingBox']
                left = ceil(width * box['Left'] / scaling_factor)
                top = ceil(height * box['Top'] / scaling_factor)
                right = ceil(left + (width * box['Width']) / scaling_factor)
                bottom = ceil(top + (height * box['Height']) / scaling_factor)
                words_and_boxes.append((block['Text'], [(left, top), (right, top), (right, bottom), (left, bottom)]))

        all_words_and_boxes.append((page_index, words_and_boxes))

    # Now match diagnoses terms
    for diag in diagnoses or []:
        term = (diag.get("term") or "").strip()
        if not term:
            continue

        term_lower = term.lower()
        term_words = term_lower.split()
        term_hits = []

        for page_index, words_and_boxes in all_words_and_boxes:
            words = [w.lower() for w, _ in words_and_boxes]

            # Exact match (full phrase)
            joined = " ".join(words)
            if term_lower in joined:
                start_idx = None
                for i in range(len(words) - len(term_words) + 1):
                    if words[i:i + len(term_words)] == term_words:
                        start_idx = i
                        break
                if start_idx is not None:
                    rects = [words_and_boxes[start_idx + j][1] for j in range(len(term_words))]
                    x0 = min(pt[0][0] for pt in rects)
                    y0 = min(pt[0][1] for pt in rects)
                    x1 = max(pt[2][0] for pt in rects)
                    y1 = max(pt[2][1] for pt in rects)
                    term_hits.append((page_index + 1, (x0, y0, x1, y1)))

            # Partial match if not found
            if not term_hits and allow_partial:
                for word in term_words:
                    for w, coords in words_and_boxes:
                        if word == w.lower():
                            x0, y0 = coords[0]
                            x1, y1 = coords[2]
                            term_hits.append((page_index + 1, (x0, y0, x1, y1)))

        if term_hits:
            results[term] = term_hits
        else:
            print(f"[INFO] No OCR matches found for: {term}")

    return results

def find_term_coordinates_with_ocr_v2(pdf_path, diagnoses, allow_partial=True):
    """
    Performs OCR on the PDF using AWS Textract and finds coordinates for terms.
    - Prefers full phrase match for the term.
    - If full match not found and allow_partial=True, merges all matched words into one bounding box.
    Works for scanned PDFs (image-based).
    """
    results = {}

    # Convert PDF pages to images for Textract
    target_dpi = 144
    pdf_pages = convert_from_path(pdf_path, dpi=target_dpi)
    textract_dpi = 72
    scaling_factor = target_dpi / textract_dpi

    all_words_and_boxes = []

    for page_index, page in enumerate(pdf_pages):
        width, height = page.size
        byte_arr = BytesIO()
        page.save(byte_arr, format='PNG')
        image_bytes = byte_arr.getvalue()

        # OCR using Textract
        response = textract_client.analyze_document(
            Document={'Bytes': image_bytes},
            FeatureTypes=['LAYOUT']
        )

        words_and_boxes = []
        for block in response['Blocks']:
            if block['BlockType'] == "WORD":
                box = block['Geometry']['BoundingBox']
                left = ceil(width * box['Left'] / scaling_factor)
                top = ceil(height * box['Top'] / scaling_factor)
                right = ceil(left + (width * box['Width']) / scaling_factor)
                bottom = ceil(top + (height * box['Height']) / scaling_factor)
                words_and_boxes.append((block['Text'], [
                    (left, top), (right, top), (right, bottom), (left, bottom)
                ]))

        all_words_and_boxes.append((page_index, words_and_boxes))

    # Match terms
    for diag in diagnoses or []:
        term = (diag.get("term") or "").strip()
        if not term:
            continue

        term_lower = term.lower()
        term_words = term_lower.split()
        term_hits = []

        for page_index, words_and_boxes in all_words_and_boxes:
            words = [w.lower() for w, _ in words_and_boxes]

            # Exact match (full phrase)
            joined = " ".join(words)
            if term_lower in joined:
                start_idx = None
                for i in range(len(words) - len(term_words) + 1):
                    if words[i:i + len(term_words)] == term_words:
                        start_idx = i
                        break
                if start_idx is not None:
                    rects = [words_and_boxes[start_idx + j][1] for j in range(len(term_words))]
                    x0 = min(r[0][0] for r in rects)
                    y0 = min(r[0][1] for r in rects)
                    x1 = max(r[2][0] for r in rects)
                    y1 = max(r[2][1] for r in rects)
                    term_hits.append((page_index + 1, (x0, y0, x1, y1)))

            # Fallback: Merge all matching words if full phrase not found
            if not term_hits and allow_partial:
                matched_boxes = []
                for word in term_words:
                    for w, coords in words_and_boxes:
                        if word == w.lower():
                            matched_boxes.append(coords)

                if matched_boxes:
                    x0 = min(c[0][0] for c in matched_boxes)
                    y0 = min(c[0][1] for c in matched_boxes)
                    x1 = max(c[2][0] for c in matched_boxes)
                    y1 = max(c[2][1] for c in matched_boxes)
                    term_hits.append((page_index + 1, (x0, y0, x1, y1)))

        if term_hits:
            results[term] = term_hits
        else:
            print(f"[INFO] No OCR matches found for: {term}")

    return results

def find_term_coordinates_with_ocr_v3(pdf_path, diagnoses, allow_partial=True, merge_threshold=50):
    """
    Performs OCR on the PDF using AWS Textract and finds term coordinates.
    Improvements:
    - Prefers full phrase match.
    - If full match not found, merges nearby partial matches into one bounding box
      (only if words are within merge_threshold pixels vertically).
    - Works for scanned PDFs.
    """
    results = {}

    # Convert PDF pages to images for Textract
    target_dpi = 144
    pdf_pages = convert_from_path(pdf_path, dpi=target_dpi)
    textract_dpi = 72
    scaling_factor = target_dpi / textract_dpi

    all_words_and_boxes = []

    for page_index, page in enumerate(pdf_pages):
        width, height = page.size
        byte_arr = BytesIO()
        page.save(byte_arr, format='PNG')
        image_bytes = byte_arr.getvalue()

        # Run Textract OCR
        response = textract_client.analyze_document(
            Document={'Bytes': image_bytes},
            FeatureTypes=['LAYOUT']
        )

        words_and_boxes = []
        for block in response['Blocks']:
            if block['BlockType'] == "WORD":
                box = block['Geometry']['BoundingBox']
                left = ceil(width * box['Left'] / scaling_factor)
                top = ceil(height * box['Top'] / scaling_factor)
                right = ceil(left + (width * box['Width']) / scaling_factor)
                bottom = ceil(top + (height * box['Height']) / scaling_factor)
                words_and_boxes.append((block['Text'], [
                    (left, top), (right, top), (right, bottom), (left, bottom)
                ]))

        all_words_and_boxes.append((page_index, words_and_boxes))

    # Process each diagnosis term
    for diag in diagnoses or []:
        term = (diag.get("term") or "").strip()
        if not term:
            continue

        term_lower = term.lower()
        term_words = term_lower.split()
        term_hits = []

        for page_index, words_and_boxes in all_words_and_boxes:
            words = [w.lower() for w, _ in words_and_boxes]

            # Exact phrase match
            joined = " ".join(words)
            if term_lower in joined:
                start_idx = None
                for i in range(len(words) - len(term_words) + 1):
                    if words[i:i + len(term_words)] == term_words:
                        start_idx = i
                        break
                if start_idx is not None:
                    rects = [words_and_boxes[start_idx + j][1] for j in range(len(term_words))]
                    x0 = min(r[0][0] for r in rects)
                    y0 = min(r[0][1] for r in rects)
                    x1 = max(r[2][0] for r in rects)
                    y1 = max(r[2][1] for r in rects)
                    term_hits.append((page_index + 1, (x0, y0, x1, y1)))
                    continue  # phrase match done, skip fallback

            # Fallback: Partial match with merging
            if allow_partial:
                matched_boxes = []
                for word in term_words:
                    for w, coords in words_and_boxes:
                        if word == w.lower():
                            matched_boxes.append(coords)

                if matched_boxes:
                    # Sort by vertical (y) then horizontal (x)
                    matched_boxes.sort(key=lambda c: (c[0][1], c[0][0]))

                    # Group words that are close vertically
                    groups = []
                    current_group = [matched_boxes[0]]
                    for i in range(1, len(matched_boxes)):
                        prev_box = matched_boxes[i - 1]
                        curr_box = matched_boxes[i]
                        prev_y = prev_box[0][1]
                        curr_y = curr_box[0][1]

                        if abs(curr_y - prev_y) <= merge_threshold:
                            current_group.append(curr_box)
                        else:
                            groups.append(current_group)
                            current_group = [curr_box]
                    groups.append(current_group)

                    # Merge each group into a single bounding box
                    for group in groups:
                        x0 = min(c[0][0] for c in group)
                        y0 = min(c[0][1] for c in group)
                        x1 = max(c[2][0] for c in group)
                        y1 = max(c[2][1] for c in group)
                        term_hits.append((page_index + 1, (x0, y0, x1, y1)))

        if term_hits:
            results[term] = term_hits
        else:
            print(f"[INFO] No OCR matches found for: {term}")

    return results

def find_term_coordinates_with_ocr_v4(pdf_path, diagnoses, allow_partial=True, max_vertical_gap=50, max_horizontal_gap=100):
    """
    Performs OCR on the PDF using AWS Textract and finds term coordinates.
    Improvements:
    - Prefers full phrase match first.
    - If full match not found, merges nearby partial matches into groups
      using BOTH vertical and horizontal distance thresholds.
    - Avoids merging across paragraphs or large gaps.
    """
    results = {}

    # Convert PDF pages to images for Textract
    target_dpi = 144
    pdf_pages = convert_from_path(pdf_path, dpi=target_dpi)
    textract_dpi = 72
    scaling_factor = target_dpi / textract_dpi

    all_words_and_boxes = []

    for page_index, page in enumerate(pdf_pages):
        width, height = page.size
        byte_arr = BytesIO()
        page.save(byte_arr, format='PNG')
        image_bytes = byte_arr.getvalue()

        # Run Textract OCR
        response = textract_client.analyze_document(
            Document={'Bytes': image_bytes},
            FeatureTypes=['LAYOUT']
        )

        words_and_boxes = []
        for block in response['Blocks']:
            if block['BlockType'] == "WORD":
                box = block['Geometry']['BoundingBox']
                left = ceil(width * box['Left'] / scaling_factor)
                top = ceil(height * box['Top'] / scaling_factor)
                right = ceil(left + (width * box['Width']) / scaling_factor)
                bottom = ceil(top + (height * box['Height']) / scaling_factor)
                words_and_boxes.append((block['Text'], [
                    (left, top), (right, top), (right, bottom), (left, bottom)
                ]))

        all_words_and_boxes.append((page_index, words_and_boxes))

    # Process each diagnosis term
    for diag in diagnoses or []:
        term = (diag.get("term") or "").strip()
        if not term:
            continue

        term_lower = term.lower()
        term_words = term_lower.split()
        term_hits = []

        for page_index, words_and_boxes in all_words_and_boxes:
            words = [w.lower() for w, _ in words_and_boxes]

            # Exact phrase match first
            joined = " ".join(words)
            if term_lower in joined:
                start_idx = None
                for i in range(len(words) - len(term_words) + 1):
                    if words[i:i + len(term_words)] == term_words:
                        start_idx = i
                        break
                if start_idx is not None:
                    rects = [words_and_boxes[start_idx + j][1] for j in range(len(term_words))]
                    x0 = min(r[0][0] for r in rects)
                    y0 = min(r[0][1] for r in rects)
                    x1 = max(r[2][0] for r in rects)
                    y1 = max(r[2][1] for r in rects)
                    term_hits.append((page_index + 1, (x0, y0, x1, y1)))
                    continue  # phrase match found, skip fallback

            # Fallback: Partial match with smarter merging
            if allow_partial:
                matched_boxes = []
                for word in term_words:
                    for w, coords in words_and_boxes:
                        if word == w.lower():
                            matched_boxes.append(coords)

                if matched_boxes:
                    # Sort by vertical (y) then horizontal (x)
                    matched_boxes.sort(key=lambda c: (c[0][1], c[0][0]))

                    groups = []
                    current_group = [matched_boxes[0]]

                    for i in range(1, len(matched_boxes)):
                        prev_box = current_group[-1]
                        curr_box = matched_boxes[i]

                        # Calculate gaps
                        vertical_gap = abs(curr_box[0][1] - prev_box[0][1])
                        horizontal_gap = abs(curr_box[0][0] - prev_box[2][0])

                        if vertical_gap <= max_vertical_gap and horizontal_gap <= max_horizontal_gap:
                            current_group.append(curr_box)
                        else:
                            groups.append(current_group)
                            current_group = [curr_box]

                    groups.append(current_group)

                    # Merge each group into a single bounding box
                    for group in groups:
                        x0 = min(c[0][0] for c in group)
                        y0 = min(c[0][1] for c in group)
                        x1 = max(c[2][0] for c in group)
                        y1 = max(c[2][1] for c in group)
                        term_hits.append((page_index + 1, (x0, y0, x1, y1)))

        if term_hits:
            results[term] = term_hits
        else:
            print(f"[INFO] No OCR matches found for: {term}")

    return results

def find_term_coordinates_with_ocr_v5(pdf_path, diagnoses, allow_partial=True):
    """
    Performs OCR on the PDF using AWS Textract and finds term coordinates.
    Improvements:
    - Full phrase match using contiguous OCR words.
    - Does NOT merge unrelated words.
    - If full match not found, highlights individual words (partial match).
    - Handles punctuation and brackets by normalizing OCR text.
    """
    results = {}

    # Convert PDF pages to images for Textract
    target_dpi = 144
    pdf_pages = convert_from_path(pdf_path, dpi=target_dpi)
    textract_dpi = 72
    scaling_factor = target_dpi / textract_dpi

    all_words_and_boxes = []

    for page_index, page in enumerate(pdf_pages):
        width, height = page.size
        byte_arr = BytesIO()
        page.save(byte_arr, format='PNG')
        image_bytes = byte_arr.getvalue()

        # Run Textract OCR
        response = textract_client.analyze_document(
            Document={'Bytes': image_bytes},
            FeatureTypes=['LAYOUT']
        )

        words_and_boxes = []
        for block in response['Blocks']:
            if block['BlockType'] == "WORD":
                text = block['Text']
                box = block['Geometry']['BoundingBox']
                # Textract returns normalized [0,1] bbox; map directly to pixel space
                left = int(round(box['Left'] * width))
                top = int(round(box['Top'] * height))
                right = int(round((box['Left'] + box['Width']) * width))
                bottom = int(round((box['Top'] + box['Height']) * height))
                words_and_boxes.append((text, [
                    (left, top), (right, top), (right, bottom), (left, bottom)
                ]))

        all_words_and_boxes.append((page_index, words_and_boxes))

    # Process each diagnosis term
    for diag in diagnoses or []:
        term = (diag.get("term") or "").strip()
        print(f"[INFO] Processing term: {term}")
        if not term:
            continue

        # Normalize term for matching: lowercase and strip ALL punctuation consistently
        term_clean = re.sub(r'[^\w\s]', ' ', term).lower()
        term_words = [t for t in term_clean.split() if t]
        print(f"[INFO] Normalized term words: {term_words}")
        if not term_words:
            continue
        term_hits = []

        for page_index, words_and_boxes in all_words_and_boxes:
            # Normalize OCR words the same way as the term (lowercase + strip punctuation)
            words_norm = [re.sub(r'[^\w\s]', ' ', w).lower() for w, _ in words_and_boxes]

            i = 0
            while i < len(words_norm):
                if words_norm[i] == term_words[0]:
                    # Possible start of phrase (contiguous match)
                    match_coords = [words_and_boxes[i][1]]
                    j = 1
                    while j < len(term_words) and i + j < len(words_norm) and words_norm[i + j] == term_words[j]:
                        match_coords.append(words_and_boxes[i + j][1])
                        j += 1

                    if j == len(term_words):
                        # Full phrase matched contiguously
                        x0 = min(c[0][0] for c in match_coords)
                        y0 = min(c[0][1] for c in match_coords)
                        x1 = max(c[2][0] for c in match_coords)
                        y1 = max(c[2][1] for c in match_coords)
                        term_hits.append((page_index + 1, (x0, y0, x1, y1)))
                        i += j
                        continue

                i += 1

            # Fallback: Partial match if phrase not found on any page
            if not term_hits and allow_partial:
                for idx, (w, coords) in enumerate(words_and_boxes):
                    w_norm = words_norm[idx]
                    if w_norm and any(SequenceMatcher(None, w_norm, tw).ratio() >= sim_threshold for tw in term_words):
                        x0, y0 = coords[0]
                        x1, y1 = coords[2]
                        term_hits.append((page_index + 1, (x0, y0, x1, y1)))
        
        
        print(f"[INFO] OCR matches found for: {term} is {term_hits}")
        if term_hits:
            results[term] = term_hits
        else:
            print(f"[INFO] No OCR matches found for: {term}")

    return results


def find_term_coordinates_with_ocr_v6(pdf_path, diagnoses, allow_partial=True, max_gap=2):
    """
    Performs OCR on the PDF using AWS Textract and finds term coordinates.
    Improvements:
    - Handles multi-line matches for phrases.
    - Allows small gaps (max_gap words) between phrase words.
    - Full phrase match preferred; partial fallback if not possible.
    """
    results = {}

    # Convert PDF pages to images for Textract
    target_dpi = 144
    pdf_pages = convert_from_path(pdf_path, dpi=target_dpi)
    textract_dpi = 72
    scaling_factor = target_dpi / textract_dpi

    all_words_and_boxes = []

    for page_index, page in enumerate(pdf_pages):
        width, height = page.size
        byte_arr = BytesIO()
        page.save(byte_arr, format='PNG')
        image_bytes = byte_arr.getvalue()

        # Run Textract OCR
        response = textract_client.analyze_document(
            Document={'Bytes': image_bytes},
            FeatureTypes=['LAYOUT']
        )

        words_and_boxes = []
        for block in response['Blocks']:
            if block['BlockType'] == "WORD":
                text = block['Text']
                box = block['Geometry']['BoundingBox']
                # Textract returns normalized [0,1] coordinates; convert to pixels directly
                left = int(round(box['Left'] * width))
                top = int(round(box['Top'] * height))
                right = int(round((box['Left'] + box['Width']) * width))
                bottom = int(round((box['Top'] + box['Height']) * height))
                words_and_boxes.append((text, [
                    (left, top), (right, top), (right, bottom), (left, bottom)
                ]))

        all_words_and_boxes.append((page_index, words_and_boxes))

    # Process each diagnosis term
    for diag in diagnoses or []:
        term = (diag.get("term") or "").strip()
        print(f"[INFO] Processing term: {term}")
        if not term:
            continue

        term_clean = re.sub(r'[^\w\s()]', '', term)
        term_words = term_clean.split()
        term_hits = []

        for page_index, words_and_boxes in all_words_and_boxes:
            words = [re.sub(r'[^\w\s()]', '', w) for w, _ in words_and_boxes]

            i = 0
            while i < len(words):
                if words[i] == term_words[0]:
                    # Possible start of phrase
                    match_coords = [words_and_boxes[i][1]]
                    j = 1
                    k = i + 1
                    gap = 0

                    while j < len(term_words) and k < len(words):
                        if words[k] == term_words[j]:
                            match_coords.append(words_and_boxes[k][1])
                            j += 1
                            gap = 0
                        else:
                            gap += 1
                            if gap > max_gap:
                                break
                        k += 1

                    if j == len(term_words):
                        # Full phrase matched across lines/gaps
                        x0 = min(c[0][0] for c in match_coords)
                        y0 = min(c[0][1] for c in match_coords)
                        x1 = max(c[2][0] for c in match_coords)
                        y1 = max(c[2][1] for c in match_coords)
                        term_hits.append((page_index + 1, (x0, y0, x1, y1)))
                        i = k
                        continue

                i += 1

            # Fallback: partial word matches
            if not term_hits and allow_partial:
                for w, coords in words_and_boxes:
                    if w in [tw for tw in term_words]:
                        x0, y0 = coords[0]
                        x1, y1 = coords[2]
                        term_hits.append((page_index + 1, (x0, y0, x1, y1)))

        if term_hits:
            results[term] = term_hits
        else:
            print(f"[INFO] No OCR matches found for: {term}")

    return results

from difflib import SequenceMatcher

def find_term_coordinates_with_ocr_v7(pdf_path, diagnoses, allow_partial=True, max_gap=3, sim_threshold=0.82):
    """
    Handles multi-line phrase matching and creates separate bounding boxes for each term.
    Prevents overlapping highlights by tracking matched word positions.
    """
    results = {}

    target_dpi = 144
    pdf_pages = convert_from_path(pdf_path, dpi=target_dpi)
    textract_dpi = 72
    scaling_factor = target_dpi / textract_dpi

    all_words_and_boxes = []

    for page_index, page in enumerate(pdf_pages):
        width, height = page.size
        byte_arr = BytesIO()
        page.save(byte_arr, format='PNG')
        image_bytes = byte_arr.getvalue()

        response = textract_client.analyze_document(
            Document={'Bytes': image_bytes},
            FeatureTypes=['LAYOUT']
        )

        words_and_boxes = []
        for block in response['Blocks']:
            if block['BlockType'] == "WORD":
                text = block['Text']
                box = block['Geometry']['BoundingBox']
                left = ceil(width * box['Left'] / scaling_factor)
                top = ceil(height * box['Top'] / scaling_factor)
                right = ceil(left + (width * box['Width']) / scaling_factor)
                bottom = ceil(top + (height * box['Height']) / scaling_factor)
                words_and_boxes.append((text, [
                    (left, top), (right, top), (right, bottom), (left, bottom)
                ]))

        all_words_and_boxes.append((page_index, words_and_boxes))

    def normalize_text(text):
        """Improved normalization for medical terminology - preserve case and important punctuation"""
        # Only replace hyphens and commas with spaces, preserve parentheses and colons
        text = re.sub(r'[\-–—,;]', ' ', text)
        # Remove only unnecessary punctuation, keep parentheses, colons, and apostrophes
        text = re.sub(r'[^\w\s\'\(\):]', ' ', text)
        # Normalize whitespace but preserve original case
        text = re.sub(r'\s+', ' ', text.strip())
        return text

    # Track all matches to prevent overlaps
    all_matches = []

    for diag in diagnoses or []:
        term = (diag.get("term") or "").strip()
        print(f"[INFO] Processing term: {term}")
        if not term:
            continue

        # Improved normalization
        term_clean = normalize_text(term)
        term_words = [t for t in term_clean.split() if t]
        if not term_words:
            continue
        
        print(f"[DEBUG] Normalized term words: {term_words}")
        term_hits = []

        for page_index, words_and_boxes in all_words_and_boxes:
            # Normalize OCR words using same function
            words_norm = [normalize_text(w) for w, _ in words_and_boxes]
            
            i = 0
            while i < len(words_norm):
                # Check if this position was already matched by a previous term
                already_matched = any(
                    match_page == page_index + 1 and 
                    i >= match_start and i <= match_end
                    for _, match_page, match_start, match_end in all_matches
                )
                
                if already_matched:
                    i += 1
                    continue

                # fuzzy compare for first token (case-insensitive comparison)
                if words_norm[i] and SequenceMatcher(None, words_norm[i].lower(), term_words[0].lower()).ratio() >= sim_threshold:
                    matched_indices = [i]  # Track which word indices were matched
                    start_idx = i
                    j, k, gap = 1, i + 1, 0

                    while j < len(term_words) and k < len(words_norm):
                        if words_norm[k] and SequenceMatcher(None, words_norm[k].lower(), term_words[j].lower()).ratio() >= sim_threshold:
                            matched_indices.append(k)
                            j += 1
                            gap = 0
                        else:
                            gap += 1
                            if gap > max_gap:
                                break
                        k += 1

                    if j == len(term_words):
                        # Create bounding box only for the matched words
                        match_boxes = [words_and_boxes[idx][1] for idx in matched_indices]
                        all_points = [pt for box in match_boxes for pt in box]
                        x0 = min(p[0] for p in all_points)
                        y0 = min(p[1] for p in all_points)
                        x1 = max(p[0] for p in all_points)
                        y1 = max(p[1] for p in all_points)
                        
                        term_hits.append((page_index + 1, (x0, y0, x1, y1)))
                        
                        # Record this match to prevent overlaps
                        end_idx = max(matched_indices)
                        all_matches.append((term, page_index + 1, start_idx, end_idx))
                        
                        print(f"[DEBUG] Found match for '{term}' at page {page_index + 1}, indices {matched_indices}")
                        i = k
                        continue

                i += 1

            # Only try partial matching if no full matches found
            if not term_hits and allow_partial:
                print(f"[DEBUG] Trying partial matching for: {term}")
                term_word_set = set(term_words)
                for idx, (w, coords) in enumerate(words_and_boxes):
                    # Check if this word position was already matched
                    already_matched = any(
                        match_page == page_index + 1 and 
                        idx >= match_start and idx <= match_end
                        for _, match_page, match_start, match_end in all_matches
                    )
                    
                    if already_matched:
                        continue
                        
                    w_norm = words_norm[idx]
                    if w_norm and w_norm in term_word_set:
                        x0, y0 = coords[0]
                        x1, y1 = coords[2]
                        term_hits.append((page_index + 1, (x0, y0, x1, y1)))
                        # Record partial match
                        all_matches.append((term, page_index + 1, idx, idx))

        if term_hits:
            results[term] = term_hits
        else:
            print(f"[INFO] No OCR matches found for: {term}")

    return results

def find_term_coordinates_substring_approach(pdf_path, diagnoses, sim_threshold=0.75):
    """
    Simple approach with extensive debugging to find what's going wrong.
    """
    results = {}

    target_dpi = 144
    pdf_pages = convert_from_path(pdf_path, dpi=target_dpi)
    textract_dpi = 72
    scaling_factor = target_dpi / textract_dpi

    all_words_and_boxes = []

    for page_index, page in enumerate(pdf_pages):
        width, height = page.size
        byte_arr = BytesIO()
        page.save(byte_arr, format='PNG')
        image_bytes = byte_arr.getvalue()

        response = textract_client.analyze_document(
            Document={'Bytes': image_bytes},
            FeatureTypes=['LAYOUT']
        )

        words_and_boxes = []
        for block in response['Blocks']:
            if block['BlockType'] == "WORD":
                text = block['Text']
                box = block['Geometry']['BoundingBox']
                left = ceil(width * box['Left'] / scaling_factor)
                top = ceil(height * box['Top'] / scaling_factor)
                right = ceil(left + (width * box['Width']) / scaling_factor)
                bottom = ceil(top + (height * box['Height']) / scaling_factor)
                words_and_boxes.append((text, [
                    (left, top), (right, top), (right, bottom), (left, bottom)
                ]))

        all_words_and_boxes.append((page_index, words_and_boxes))
        
        # DEBUG: Print all OCR words found on this page
        print(f"\n[DEBUG] Page {page_index + 1} OCR Words:")
        for i, (word, coords) in enumerate(words_and_boxes):
            print(f"  {i}: '{word}' at {coords[0]} to {coords[2]}")

    for diag in diagnoses or []:
        term = (diag.get("term") or "").strip()
        print(f"\n[INFO] ===== Processing term: '{term}' =====")
        if not term:
            continue

        term_hits = []
        
        # Try multiple matching strategies
        for page_index, words_and_boxes in all_words_and_boxes:
            print(f"\n[DEBUG] Searching on page {page_index + 1}")
            
            # Strategy 1: Simple word-by-word exact matching
            term_words = term.lower().split()
            print(f"[DEBUG] Looking for words: {term_words}")
            
            for i, (word, coords) in enumerate(words_and_boxes):
                word_lower = word.lower()
                print(f"[DEBUG] word_lower: '{word_lower}'")
                
                # Check if this word starts our term
                if word_lower in term_words[0] or term_words[0] in word_lower:
                    print(f"[DEBUG] Potential start match: '{word}' matches '{term_words[0]}'")
                    
                    # Try to match the complete term starting from this position
                    matched_coords = []
                    matched_words = []
                    j = 0  # term word index
                    k = i  # OCR word index
                    
                    while j < len(term_words) and k < len(words_and_boxes):
                        ocr_word = words_and_boxes[k][0].lower()
                        search_word = term_words[j]
                        
                        # Flexible matching
                        if (ocr_word == search_word or 
                            search_word in ocr_word or 
                            ocr_word in search_word or
                            SequenceMatcher(None, ocr_word, search_word).ratio() > 0.7):
                            
                            matched_coords.append(words_and_boxes[k][1])
                            matched_words.append(words_and_boxes[k][0])
                            j += 1
                            print(f"[DEBUG] Matched: '{ocr_word}' -> '{search_word}'")
                        
                        k += 1
                        
                        # Stop if we've gone too far without finding the next word
                        if k - i > len(term_words) + 3:
                            break
                    
                    # If we matched all words in the term
                    if j == len(term_words):
                        print(f"[DEBUG] ✓ Complete match found: {matched_words}")
                        
                        # Create bounding box
                        all_points = [pt for coords in matched_coords for pt in coords]
                        x0 = min(p[0] for p in all_points)
                        y0 = min(p[1] for p in all_points)
                        x1 = max(p[0] for p in all_points)
                        y1 = max(p[1] for p in all_points)
                        
                        term_hits.append((page_index + 1, (x0, y0, x1, y1)))
                        print(f"[DEBUG] Bounding box: ({x0}, {y0}, {x1}, {y1})")
                        break
            
            # Strategy 2: If no exact match, try partial matching
            if not term_hits:
                print(f"[DEBUG] No exact match found, trying partial matching...")
                
                for word_in_term in term_words:
                    for i, (word, coords) in enumerate(words_and_boxes):
                        word_lower = word.lower()
                        if (word_lower == word_in_term or 
                            word_in_term in word_lower or 
                            SequenceMatcher(None, word_lower, word_in_term).ratio() > 0.8):
                            
                            print(f"[DEBUG] Partial match: '{word}' matches '{word_in_term}'")
                            x0, y0 = coords[0]
                            x1, y1 = coords[2]
                            term_hits.append((page_index + 1, (x0, y0, x1, y1)))

        if term_hits:
            results[term] = term_hits
            print(f"[INFO] ✓ Found {len(term_hits)} matches for '{term}'")
        else:
            print(f"[INFO] ✗ No matches found for '{term}'")

    print(f"\n[FINAL] Results: {list(results.keys())}")
    return results


def find_matching_word_coords(term, line_words):
    """Find coordinates of words that match the search term"""
    term_words = term.lower().split()
    line_text_words = [word.lower() for word, _ in line_words]
    
    # Find the starting position of the term in the line
    for i in range(len(line_text_words) - len(term_words) + 1):
        if all(term_words[j] in line_text_words[i + j] or 
               SequenceMatcher(None, term_words[j], line_text_words[i + j]).ratio() > 0.8
               for j in range(len(term_words))):
            # Return coordinates for the matched words
            return [line_words[i + j][1] for j in range(len(term_words))]
    
    return []


def find_fuzzy_substring_coords(term, line_words, threshold=0.8):
    """Find coordinates using fuzzy matching for substrings"""
    term_lower = term.lower()
    
    # Try different word combinations
    for start_idx in range(len(line_words)):
        for end_idx in range(start_idx + 1, len(line_words) + 1):
            word_sequence = ' '.join([word for word, _ in line_words[start_idx:end_idx]])
            
            if SequenceMatcher(None, word_sequence.lower(), term_lower).ratio() >= threshold:
                return [coords for _, coords in line_words[start_idx:end_idx]]
    
    return []

# def normalize_text(text):
#     """Normalize text by lowering, removing extra spaces and punctuation."""
#     text = text.lower()
#     text = re.sub(r'\s+', ' ', text)
#     return text.strip()

def find_term_coordinates_with_ocr(pdf_path, diagnoses, allow_partial=True):
    """
    Performs OCR on the PDF using AWS Textract and finds coordinates for full terms (no splitting).
    Works even if PDF has no text layer (scanned PDF).
    Uses normalization to handle extra spaces or line breaks.
    """
    results = {}

    # Convert PDF pages to images for Textract
    target_dpi = 144
    pdf_pages = convert_from_path(pdf_path, dpi=target_dpi)
    textract_dpi = 72
    scaling_factor = target_dpi / textract_dpi

    all_words_and_boxes = []

    for page_index, page in enumerate(pdf_pages):
        width, height = page.size
        byte_arr = BytesIO()
        page.save(byte_arr, format='PNG')
        image_bytes = byte_arr.getvalue()

        # Perform OCR using Textract
        response = textract_client.analyze_document(
            Document={'Bytes': image_bytes},
            FeatureTypes=['LAYOUT']
        )

        words_and_boxes = []
        for block in response['Blocks']:
            if block['BlockType'] == "WORD":
                box = block['Geometry']['BoundingBox']
                left = ceil(width * box['Left'] / scaling_factor)
                top = ceil(height * box['Top'] / scaling_factor)
                right = ceil(left + (width * box['Width']) / scaling_factor)
                bottom = ceil(top + (height * box['Height']) / scaling_factor)
                words_and_boxes.append((block['Text'], [
                    (left, top), (right, top), (right, bottom), (left, bottom)
                ]))

        all_words_and_boxes.append((page_index, words_and_boxes))

        # Debug: print OCR text of the page
        page_text = " ".join([w for w, _ in words_and_boxes])
        print(f"[DEBUG] Page {page_index+1} OCR text: {page_text}")

    # Match terms
    for diag in diagnoses or []:
        term = (diag.get("term") or "").strip()
        if not term:
            continue

        # term_lower = normalize_text(term)
        term_lower = term.lower()
        print(f"[DEBUG] Searching for term: {term_lower}")
        term_hits = []

        for page_index, words_and_boxes in all_words_and_boxes:
            words = [w for w, _ in words_and_boxes]
            normalized_page_text = normalize_text(" ".join(words))

            if term_lower in normalized_page_text:
                # Collect bounding boxes for words that are part of the term
                term_parts = term_lower.split()
                words_lower = [w.lower() for w in words]

                match_start = None
                for i in range(len(words_lower) - len(term_parts) + 1):
                    if words_lower[i:i + len(term_parts)] == term_parts:
                        match_start = i
                        break

                if match_start is not None:
                    rects = [words_and_boxes[match_start + j][1] for j in range(len(term_parts))]
                    # Merge rects into one bounding box
                    x0 = min(coords[0][0] for coords in rects)
                    y0 = min(coords[0][1] for coords in rects)
                    x1 = max(coords[2][0] for coords in rects)
                    y1 = max(coords[2][1] for coords in rects)
                    term_hits.append((page_index + 1, (x0, y0, x1, y1)))

            # Partial fallback if no full match
            if allow_partial and not term_hits:
                for w, coords in words_and_boxes:
                    if term_lower in w.lower():
                        x0, y0 = coords[0]
                        x1, y1 = coords[2]
                        term_hits.append((page_index + 1, (x0, y0, x1, y1)))

        if term_hits:
            results[term] = term_hits
        else:
            print(f"[INFO] No OCR matches found for: {term}")

    return results

def highlight_terms_with_ocr_coordinates(pdf_path, output_pdf_path, hits, diagnoses):
    """
    Highlights terms on the original PDF using OCR-based coordinates.

    Args:
        pdf_path (str): Original PDF file path
        output_pdf_path (str): Path to save highlighted PDF
        hits (dict): {term: [(page_num, (x0, y0, x1, y1)), ...]} from find_term_coordinates_with_ocr
        diagnoses (list): List of dicts with 'term' and 'snomed_code'
    """
    doc = fitz.open(pdf_path)

    # Create a lookup for SNOMED codes
    snomed_map = {d.get("term"): d.get("snomed_code", "") for d in diagnoses}

    for term, coords_list in hits.items():
        snomed_code = snomed_map.get(term, "Not available")
        for page_num, (x0, y0, x1, y1) in coords_list:
            page = doc[page_num - 1]
            rect = fitz.Rect(x0, y0, x1, y1)

            highlight = page.add_highlight_annot(rect)
            highlight.set_colors(stroke=(1, 1, 0))  # Black
            highlight.update()

            info = {
                "title": term,
                "subject": "Diagnosis",
                "content": f"SNOMED Code: {snomed_code}"
            }
            highlight.set_info(info)

    doc.save(output_pdf_path)
    doc.close()
    print(f"[INFO] Highlighted PDF saved at: {output_pdf_path}")

def highlight_terms_in_pdf(pdf_path, output_path, hits, diagnoses):
    """
    Highlights terms and adds SNOMED codes as annotations in the PDF.
    """
    doc = fitz.open(pdf_path)

    for diag in diagnoses:
        term = diag.get("term")
        snomed_code = diag.get("snomed_code") or "N/A"
        positions = hits.get(term, [])

        for page_num, (x0, y0, x1, y1) in positions:
            page = doc[page_num - 1]
            rect = fitz.Rect(x0, y0, x1, y1)

            # Highlight the term
            highlight = page.add_highlight_annot(rect)

            # Add annotation with SNOMED code
            note_text = f"Term: {term}\nSNOMED: {snomed_code}"
            highlight.set_info(info={"title": "SNOMED Mapping"})
            highlight.set_info(note_text)
            highlight.update()

    doc.save(output_path)
    print(f"[SUCCESS] Highlighted PDF saved at: {output_path}")


def highlight_terms_with_ocr_coordinates_v2(pdf_path, output_pdf_path, hits, diagnoses):
    """
    Highlights terms on the original PDF using OCR-based coordinates with different colors for each term.

    Args:
        pdf_path (str): Original PDF file path
        output_pdf_path (str): Path to save highlighted PDF
        hits (dict): {term: [(page_num, (x0, y0, x1, y1)), ...]} from find_term_coordinates_with_ocr
        diagnoses (list): List of dicts with 'term' and 'snomed_code'
    """
    doc = fitz.open(pdf_path)

    # Create a lookup for SNOMED codes
    snomed_map = {d.get("term"): d.get("snomed_code", "") for d in diagnoses}

    # Generate random colors for each term (in RGB 0-1 range)
    term_colors = {}
    for term in hits.keys():
        term_colors[term] = (random.random(), random.random(), random.random())  # RGB tuple

    for term, coords_list in hits.items():
        snomed_code = snomed_map.get(term, "Not available")
        color = term_colors[term]

        for page_num, (x0, y0, x1, y1) in coords_list:
            page = doc[page_num - 1]
            rect = fitz.Rect(x0, y0, x1, y1)

            highlight = page.add_highlight_annot(rect)
            highlight.set_colors(stroke=color)  # Use the unique color
            highlight.update()

            info = {
                "title": term,
                "subject": "Diagnosis",
                "content": f"SNOMED Code: {snomed_code}"
            }
            highlight.set_info(info)

    doc.save(output_pdf_path)
    doc.close()
    print(f"[INFO] Highlighted PDF saved at: {output_pdf_path}")

def highlight_terms_with_ocr_coordinates_v3(pdf_path, output_pdf_path, hits, diagnoses):
    """
    Highlights terms on the original PDF using OCR-based coordinates with fixed colors for each term.

    Args:
        pdf_path (str): Original PDF file path
        output_pdf_path (str): Path to save highlighted PDF
        hits (dict): {term: [(page_num, (x0, y0, x1, y1)), ...]} from find_term_coordinates_with_ocr
        diagnoses (list): List of dicts with 'term' and 'snomed_code'
    """
    doc = fitz.open(pdf_path)

    # Create a lookup for SNOMED codes
    snomed_map = {d.get("term"): d.get("snomed_code", "") for d in diagnoses}

    # Define a fixed color palette (RGB 0-1)
    palette = [
        (1, 1, 0),       # Yellow
        (1, 0.6, 0),     # Orange
        (0, 1, 0),       # Green
        (0, 1, 1),       # Cyan
        (1, 0, 1),       # Magenta
        (0.7, 0.7, 1),   # Light Blue
        (1, 0.7, 0.7),   # Light Red
        (0.7, 1, 0.7),   # Light Green
        (1, 0.5, 0.8),   # Pink
        (0.8, 0.8, 0.4)  # Olive
    ]

    # Assign colors from the palette in sequence
    term_colors = {}
    for idx, term in enumerate(hits.keys()):
        term_colors[term] = palette[idx % len(palette)]

    # Apply highlights
    for term, coords_list in hits.items():
        snomed_code = snomed_map.get(term, "Not available")
        color = term_colors[term]
        print(f"[INFO] Highlighting term: {term}")
        print(f"[INFO] SNOMED Code: {snomed_code}")
        print(f"[INFO] Color: {color}")

        for page_num, (x0, y0, x1, y1) in coords_list:
            page = doc[page_num - 1]
            rect = fitz.Rect(x0, y0, x1, y1)

            highlight = page.add_highlight_annot(rect)
            highlight.set_colors(stroke=color)
            highlight.update()

            info = {
                "title": term,
                "subject": "Diagnosis",
                "content": f"SNOMED Code: {snomed_code}"
            }
            highlight.set_info(info)

    doc.save(output_pdf_path)
    doc.close()
    print(f"[INFO] Highlighted PDF saved at: {output_pdf_path}")


def highlight_terms_with_ocr_coordinates_v4(pdf_path, output_pdf_path, hits, diagnoses):
    """
    Highlights terms on the original PDF using OCR-based coordinates.
    Each term uses a fixed color from the palette.
    Overlapping highlights remain separate.

    Args:
        pdf_path (str): Original PDF file path
        output_pdf_path (str): Path to save highlighted PDF
        hits (dict): {term: [(page_num, (x0, y0, x1, y1)), ...]}
        diagnoses (list): List of dicts with 'term' and 'snomed_code'
    """
    doc = fitz.open(pdf_path)

    # Fixed color palette (RGB values normalized to 0-1)
    color_palette = [
        (1, 1, 0),      # Yellow
        (1, 0.6, 0),    # Orange
        (0, 1, 0),      # Green
        (0, 1, 1),      # Cyan
        (1, 0, 1),      # Magenta
        (0.8, 0.8, 1),  # Light Blue
        (1, 0.8, 0.8),  # Light Red
    ]

    snomed_map = {d.get("term"): d.get("snomed_code", "") for d in diagnoses}

    for idx, (term, coords_list) in enumerate(hits.items()):
        color = color_palette[idx % len(color_palette)]  # Cycle through palette
        snomed_code = snomed_map.get(term, "Not available")

        print(f"[INFO] Highlighting term: {term}")
        print(f"[INFO] SNOMED Code: {snomed_code}")
        print(f"[INFO] Color: {color}")

        for page_num, (x0, y0, x1, y1) in coords_list:
            page = doc[page_num - 1]
            rect = fitz.Rect(x0, y0, x1, y1)

            highlight = page.add_highlight_annot(rect)
            highlight.set_colors(stroke=color)
            highlight.update()

            # Add SNOMED info as popup content
            highlight.set_info({
                "title": term,
                "subject": "Diagnosis",
                "content": f"SNOMED Code: {snomed_code}"
            })

    doc.save(output_pdf_path)
    doc.close()
    print(f"[INFO] Highlighted PDF saved at: {output_pdf_path}")

