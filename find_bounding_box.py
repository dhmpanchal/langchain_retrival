import re
from typing import Dict, List, Tuple
import fitz  # PyMuPDF

def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())

def _tokenize(s: str) -> List[str]:
    # Simple whitespace tokenization; keep "exact words" behavior
    return _normalize_ws(s).split(" ")

def _find_token_sequences(page_words: List[Tuple[float,float,float,float,str,int,int,int]],
                          tokens: List[str],
                          case_sensitive: bool = True) -> List[Tuple[int, int]]:
    """
    Find all (start_idx, end_idx inclusive) spans in page_words whose word-text equals tokens.
    page_words: list of (x0, y0, x1, y1, text, block_no, line_no, word_no)
    """
    if not tokens:
        return []
    seqs = []
    n = len(page_words)
    m = len(tokens)
    for i in range(0, n - m + 1):
        ok = True
        for j in range(m):
            w = page_words[i + j][4]
            a = w if case_sensitive else w.lower()
            b = tokens[j] if case_sensitive else tokens[j].lower()
            if a != b:
                ok = False
                break
        if ok:
            seqs.append((i, i + m - 1))
    return seqs

def _rect_union(words_span: List[Tuple[float,float,float,float,str,int,int,int]]):
    x0 = min(w[0] for w in words_span)
    y0 = min(w[1] for w in words_span)
    x1 = max(w[2] for w in words_span)
    y1 = max(w[3] for w in words_span)
    return x0, y0, x1, y1

def find_term_coordinates_in_pdf(
    pdf_path: str,
    diagnoses: List[Dict],
    *,
    case_sensitive: bool = True
) -> List[Dict]:
    """
    Find all coordinates for each diagnosis's 'snomed_ct_term' by first locating its 'source'
    and then matching the exact term inside those spans. If not found in any source span on a page,
    fallback to searching the term anywhere on that page.

    Args:
      pdf_path: path to the PDF.
      diagnoses: list of dicts with keys: 'snomed_ct_term' (str), 'source' (str).
      case_sensitive: whether matching should be case-sensitive (default True for "exact" match).

    Returns:
      List of dicts: {
        'page_index': int,
        'term': str,
        'source': str,
        'x0': float, 'y0': float, 'x1': float, 'y1': float
      }
    """
    results: List[Dict] = []
    doc = fitz.open(pdf_path)

    try:
        for page_index in range(len(doc)):
            page = doc[page_index]
            # Extract words as (x0, y0, x1, y1, "text", block, line, word)
            words = page.get_text("words")
            # Sort by (block, line, word) to preserve reading order
            words.sort(key=lambda w: (w[5], w[6], w[7]))

            for diag in diagnoses:
                term = (diag.get("snomed_ct_term") or "").strip()
                source = (diag.get("source") or "").strip()
                if not term or not source:
                    continue

                term_tokens = _tokenize(term if case_sensitive else term.lower())
                source_tokens = _tokenize(source if case_sensitive else source.lower())

                # Prepare words with case handling for matching
                words_for_match = []
                for (x0, y0, x1, y1, t, b, l, wn) in words:
                    words_for_match.append((
                        x0, y0, x1, y1,
                        t if case_sensitive else t.lower(),
                        b, l, wn
                    ))

                found_any_for_diag_on_page = False

                # 1) Find all source spans on this page
                source_spans = _find_token_sequences(words_for_match, source_tokens, case_sensitive=True)

                # 2) Within each source span, try to locate the exact term
                for (s_start, s_end) in source_spans:
                    span_words = words_for_match[s_start:s_end+1]
                    term_spans_relative = _find_token_sequences(span_words, term_tokens, case_sensitive=True)
                    for (ts, te) in term_spans_relative:
                        abs_start = s_start + ts
                        abs_end = s_start + te
                        coords = _rect_union(words[abs_start:abs_end+1])  # use original words for exact coords
                        results.append({
                            "page_index": page_index,
                            "term": term,
                            "source": source,
                            "x0": coords[0], "y0": coords[1],
                            "x1": coords[2], "y1": coords[3],
                        })
                        found_any_for_diag_on_page = True

                # 3) Fallback: search the term anywhere on the page (if not found inside any source span)
                if not found_any_for_diag_on_page:
                    term_spans_global = _find_token_sequences(words_for_match, term_tokens, case_sensitive=True)
                    for (gs, ge) in term_spans_global:
                        coords = _rect_union(words[gs:ge+1])
                        results.append({
                            "page_index": page_index,
                            "term": term,
                            "source": source,
                            "x0": coords[0], "y0": coords[1],
                            "x1": coords[2], "y1": coords[3],
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
        diagnoses: list of dicts with at least {'snomed_ct_term': str, 'source': str}
        snomed_codes_list: [{ term: [ {id, key, value, description}, ... ] }, ...]
        case_sensitive: whether matching should be case-sensitive
    
    Returns:
        str: path to the saved highlighted PDF
    """
    term_to_codes = build_term_to_codes(snomed_codes_list)
    hits = find_term_coordinates_in_pdf(pdf_path, diagnoses, case_sensitive=case_sensitive)

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