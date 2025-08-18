import re
from typing import Dict, List, Tuple, Optional
import fitz  # PyMuPDF
import pymupdf
import pytesseract
from PIL import Image
from collections import defaultdict

def _normalize_text(s: str) -> str:
    """Normalize text by removing extra whitespace and standardizing punctuation."""
    if not s:
        return ""
    # Replace multiple spaces with single space, remove leading/trailing spaces
    s = re.sub(r'\s+', ' ', s.strip())
    # Standardize common punctuation patterns
    s = re.sub(r'\s*[\-\,\:]\s*', ' ', s)
    return s.lower()

def _find_text_in_words(words: List[Tuple[float, float, float, float, str, int, int, int]], 
                      search_text: str, 
                      case_sensitive: bool = False) -> List[Dict]:
    """Find text in page words using a flexible matching approach."""
    if not search_text.strip():
        return []
        
    search_text = search_text.lower() if not case_sensitive else search_text
    search_text = _normalize_text(search_text)
    full_text = ' '.join(w[4].lower() if not case_sensitive else w[4] for w in words)
    
    results = []
    start_pos = 0
    
    while True:
        idx = full_text.find(search_text, start_pos)
        if idx == -1:
            break
            
        start_pos = idx + len(search_text)
        char_pos = 0
        start_word_idx = -1
        end_word_idx = -1
        
        for i, word in enumerate(words):
            word_text = word[4].lower() if not case_sensitive else word[4]
            word_len = len(word_text)
            
            if char_pos <= idx < char_pos + word_len + 1:
                if start_word_idx == -1:
                    start_word_idx = i
                end_word_idx = i
            elif char_pos > idx + len(search_text):
                break
                
            char_pos += len(word_text) + 1
        
        if start_word_idx != -1 and end_word_idx != -1:
            x0 = min(words[i][0] for i in range(start_word_idx, end_word_idx + 1))
            y0 = min(words[i][1] for i in range(start_word_idx, end_word_idx + 1))
            x1 = max(words[i][2] for i in range(start_word_idx, end_word_idx + 1))
            y1 = max(words[i][3] for i in range(start_word_idx, end_word_idx + 1))
            
            results.append({
                'start_idx': start_word_idx,
                'end_idx': end_word_idx,
                'coords': (x0, y0, x1, y1)
            })
    
    return results

def old_find_term_coordinates_in_pdf_v1(
    pdf_path: str,
    diagnoses: List[Dict],
    *,
    case_sensitive: bool = False
) -> List[Dict]:
    """
    Find all coordinates for each diagnosis's 'term' using flexible text matching.
    
    Args:
      pdf_path: Path to the PDF file.
      diagnoses: List of dicts with keys: 'term' (str), 'source' (str).
      case_sensitive: Whether matching should be case-sensitive.
      
    Returns:
      List of dicts with page_index, term, source, and coordinates (x0,y0,x1,y1).
    """
    results: List[Dict] = []
    doc = fitz.open(pdf_path)

    try:
        for page_index in range(len(doc)):
            page = doc[page_index]
            words = page.get_text("words")
            if not words:
                continue
                
            for diag in diagnoses:
                term = (diag.get("term") or "").strip()
                source = (diag.get("source") or "").strip()
                if not term or not source:
                    continue
                
                # Debug: Print page text to verify what we're searching in
                page_text = ' '.join(w[4] for w in words)
                print(f"\nSearching for term: '{term}' in page {page_index + 1}")
                print(f"First 200 chars of page text: {page_text[:200]}...")
                
                # Try to find the term in the page
                matches = _find_text_in_words(words, term, case_sensitive)
                print(f"Found {len(matches)} matches for term: '{term}'")
                
                for match in matches:
                    x0, y0, x1, y1 = match['coords']
                    matched_text = ' '.join(words[i][4] for i in range(match['start_idx'], match['end_idx'] + 1))
                    print(f"  - Matched: '{matched_text}' at position {match['start_idx']}-{match['end_idx']}")
                    
                    results.append({
                        "page_index": page_index,
                        "term": term,
                        "source": source,
                        "x0": x0, "y0": y0,
                        "x1": x1, "y1": y1,
                    })
                
                # If no direct term matches, try to find partial matches
                if not matches and len(term.split()) > 1:
                    # Try individual words if the term is a phrase
                    for word in term.split():
                        word_matches = _find_text_in_words(words, word, case_sensitive)
                        for match in word_matches:
                            x0, y0, x1, y1 = match['coords']
                            results.append({
                                "page_index": page_index,
                                "term": word,  # Store just the matched word
                                "source": source,
                                "x0": x0, "y0": y0,
                                "x1": x1, "y1": y1,
                            })
    finally:
        doc.close()

    return results

def _get_context_around_match(words: List[Tuple[float, float, float, float, str, int, int, int]], 
                           start_idx: int, end_idx: int, window: int = 10) -> str:
    """Get text context around a match."""
    context_start = max(0, start_idx - window)
    context_end = min(len(words), end_idx + 1 + window)
    return ' '.join(words[i][4] for i in range(context_start, context_end))

def _text_similarity(text1: str, text2: str) -> float:
    """Calculate a simple text similarity score between 0 and 1."""
    from difflib import SequenceMatcher
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

def _get_context_around_match(words: List[Tuple[float, float, float, float, str, int, int, int]], 
                           start_idx: int, end_idx: int, window: int = 10) -> str:
    """Get text context around a match."""
    context_start = max(0, start_idx - window)
    context_end = min(len(words), end_idx + 1 + window)
    return ' '.join(words[i][4] for i in range(context_start, context_end))

def _text_similarity(text1: str, text2: str) -> float:
    """Calculate a simple text similarity score between 0 and 1."""
    from difflib import SequenceMatcher
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

def get_page_text_blocks(page, use_ocr=True):
    if use_ocr:
        # Convert page to image and extract text with bounding boxes
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # Get OCR data with bounding boxes
        ocr_data = pytesseract.image_to_data(
            img, 
            output_type=pytesseract.Output.DICT,
            config='--psm 6'  # Assume a single uniform block of text
        )
        
        # Convert to list of (x0, y0, x1, y1, text, ...) format
        blocks = []
        for i in range(len(ocr_data['text'])):
            if ocr_data['text'][i].strip():  # Only include non-empty text
                x0 = ocr_data['left'][i]
                y0 = ocr_data['top'][i]
                x1 = x0 + ocr_data['width'][i]
                y1 = y0 + ocr_data['height'][i]
                text = ocr_data['text'][i]
                blocks.append((x0, y0, x1, y1, text, 0, 0, 0))  # Last 3 numbers are placeholders
        return blocks
    return page.get_text("words")

def _case_sensitive_matches_on_page(page, needle):
    """
    Case-sensitive matcher using page.get_text('words').
    Searches within each line; returns list of fitz.Rect unions.
    Note: Unlike search_for(), this simple version won't match across line breaks.
    """
    rects = []
    words = page.get_text("words")  # (x0, y0, x1, y1, word, block, line, word_no)
    # group words by (block, line)
    lines = defaultdict(list)
    for x0, y0, x1, y1, w, b, l, n in words:
        lines[(b, l)].append((n, w, pymupdf.Rect(x0, y0, x1, y1)))
    # ensure order within line
    for key in sorted(lines.keys()):
        line = sorted(lines[key], key=lambda t: t[0])
        texts = [w for _, w, _ in line]
        # Build a plain "words joined by single spaces" text and a map to word indices
        joined = " ".join(texts)
        start = 0
        while True:
            idx = joined.find(needle, start)
            if idx == -1:
                break
            # map char-span back to word-span
            # build char start offset for each word
            offsets = []
            off = 0
            for w in texts:
                offsets.append(off)
                off += len(w) + 1  # 1 for inserted space
            # find first and last word indices covered by [idx, idx+len(needle))
            end_idx = idx + len(needle)
            first = None
            last = None
            for i, off in enumerate(offsets):
                wlen = len(texts[i])
                w_start = off
                w_end = off + wlen
                # overlap?
                if w_end > idx and w_start < end_idx:
                    if first is None:
                        first = i
                    last = i
            if first is not None and last is not None:
                # union rects of covered words
                r = None
                for i in range(first, last + 1):
                    rect = line[i][2]
                    r = rect if r is None else r | rect
                rects.append(r)
            start = idx + 1
    return rects

def find_term_coordinates_in_pdf(pdf_path, diagnoses, case_sensitive=False, allow_partial=True, return_quads=False):
    """
    Returns: dict {term: [(page_number, rect), ...]}
    Notes:
      - Case-insensitive search uses page.search_for() (default: case-insensitive).
      - If case_sensitive=True, we match within single lines using word data.
      - If allow_partial=True and exact match fails, we also return matches for individual words.
    """
    results = {}
    try:
        doc = pymupdf.open(pdf_path)
    except Exception as e:
        print(f"[ERROR] Cannot open PDF: {e}")
        return results

    for diag in diagnoses or []:
        term = (diag.get("term") or "").strip()
        if not term:
            continue

        term_hits = []

        for page_num, page in enumerate(doc, start=1):
            if case_sensitive:
                # manual case-sensitive search on lines
                exact_rects = _case_sensitive_matches_on_page(page, term)
            else:
                # default PyMuPDF search is already case-insensitive
                exact_rects = page.search_for(term, quads=False)  # flags not needed

            if exact_rects:
                for r in exact_rects:
                    term_hits.append((page_num, r))
                continue

            if allow_partial:
                # fallback: search each word in the term independently (case-insensitive)
                for w in filter(None, re.split(r"\s+", term)):
                    for r in page.search_for(w, quads=False):
                        term_hits.append((page_num, r))

        if term_hits:
            results[term] = term_hits
        else:
            print(f"[INFO] No matches found for: {term}")

    doc.close()
    return results

def build_term_to_codes(snomed_codes_list):
    """
    Convert the snomed_codes_list format to a simple term -> codes mapping.
    
    Args:
        snomed_codes_list: format: [{ "<term>": [ {id, key, value, description}, ... ] }, ...]
    
    Returns:
        dict: {term: [codes]}
    """
    term_to_codes = {}
    for item in snomed_codes_list or []:
        if isinstance(item, dict):
            for term, codes in item.items():
                term_to_codes[(term or "").strip()] = codes or []
    return term_to_codes


def highlight_terms_in_pdf(pdf_path, output_pdf_path, diagnoses, snomed_codes_list, case_sensitive=True):
    """
    Highlight terms in PDF and save with annotations containing SNOMED codes.
    
    Args:
        pdf_path: input PDF path
        output_pdf_path: output PDF path with highlights
        diagnoses: list of dicts with at least {'term': str, 'source': str}
        snomed_codes_list: [{ term: [ {id, key, value, description}, ... ] }, ...]
        case_sensitive: whether matching should be case-sensitive
    
    Returns:
        str: path to the saved highlighted PDF
    """
    term_to_codes = build_term_to_codes(snomed_codes_list)
    hits = find_term_coordinates_in_pdf(pdf_path, diagnoses, case_sensitive=case_sensitive)
    print(f"Found bounding box for {hits} terms in {pdf_path}")

    doc = fitz.open(pdf_path)
    try:
        for hit in hits:
            page = doc[hit["page_index"]]
            rect = fitz.Rect(hit["x0"], hit["y0"], hit["x1"], hit["y1"])
            annot = page.add_highlight_annot(rect)

            term = hit["term"]
            codes = term_to_codes.get(term, [])

            lines = []
            for c in (codes[:10] if codes else []):  # Limit to first 10 codes
                cid = str(c.get("id") or c.get("key") or "")
                val = str(c.get("value") or "")
                desc = str(c.get("description") or "")
                if desc:
                    lines.append(f"{cid}: {val} — {desc}")
                else:
                    lines.append(f"{cid}: {val}")
            content = f"SNOMED for '{term}':\n" + ("\n".join(lines) if lines else "No codes found")

            # Set annotation info so PDF viewers show it when clicked
            annot.set_info(title="SNOMED", content=content, subject=term)
            annot.set_colors(stroke=(1, 1, 0))  # yellow highlight border
            annot.set_opacity(0.3)
            annot.update()

        doc.save(output_pdf_path, deflate=True)
    finally:
        doc.close()

    return output_pdf_path

if __name__ == "__main__":
    pdf_path = "./data/gastroscopy report 02 jun (1).pdf"
    diagnoses = [{'term': 'Hiatus hernia', 'snomed_ct_term': 'Hiatus hernia', 'source': '3 cm sliding Hiatal hernia'}, {'term': 'Oesophageal stricture', 'snomed_ct_term': 'Oesophageal stricture', 'source': 'Stricture, benign in appearance, traversable after dilation.'}]
    hits = find_term_coordinates_in_pdf(pdf_path, diagnoses, case_sensitive=False)
    print(f"Found bounding box for {hits} terms in {pdf_path}")