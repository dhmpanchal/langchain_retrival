import re
from typing import Dict, List, Tuple
import fitz  # PyMuPDF

def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())

def _tokenize(s: str) -> List[str]:
    # Simple whitespace tokenization; keep “exact words” behavior
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
      case_sensitive: whether matching should be case-sensitive (default True for “exact” match).

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