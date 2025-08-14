import re
from typing import Dict, List, Tuple, Optional
import fitz  # PyMuPDF

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

def find_term_coordinates_in_pdf(
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