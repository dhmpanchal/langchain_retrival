import json
from typing import List
from new_cc_poc import POC
from find_bounding_box import highlight_terms_in_pdf


def get_snomed_codes_for_active_diagonosis(final_active_diagnosis: dict):
    if not isinstance(final_active_diagnosis, dict):
        return []

    diagnoses = final_active_diagnosis.get('diagnoses')
    if not diagnoses:
        print(f"diagnoses key not found in {final_active_diagnosis}")
        return []

    poc = POC()
    return [
        {item.get('snomed_ct_term', ''): poc.get_snomed_codes(item.get('snomed_ct_term', ''))}
        for item in diagnoses
    ]


def clean_diagnosis_data(active_diagnosis: dict, snomed_codes: List[dict]) -> dict:
    """
    For each diagnosis, find the exact matching SNOMED CT term in snomed_codes and add its code as 'snomed_code'.
    Returns a dict with the same shape as active_diagnosis: {'diagnoses': [ ... ]}
    """
    if not isinstance(active_diagnosis, dict):
        return {'diagnoses': []}

    diagnoses = active_diagnosis.get('diagnoses')
    if not isinstance(diagnoses, list):
        return {'diagnoses': []}

    term_to_candidates = {}
    for entry in snomed_codes or []:
        if isinstance(entry, dict):
            for k, v in entry.items():
                if isinstance(v, list):
                    term_to_candidates[k] = v

    cleaned: List[dict] = []
    for diag in diagnoses:
        if not isinstance(diag, dict):
            continue
        snomed_term = (diag.get('snomed_ct_term') or '').strip()
        code_value = ''

        candidates = term_to_candidates.get(snomed_term)
        if candidates is None and snomed_term:
            for k, v in term_to_candidates.items():
                if isinstance(k, str) and k.strip().lower() == snomed_term.lower():
                    candidates = v
                    break

        if candidates:
            chosen = None
            for c in candidates:
                val = str((c or {}).get('value') or '').strip()
                if val.lower() == snomed_term.lower():
                    chosen = c
                    break
            if chosen is None:
                chosen = candidates[0]
            code_value = str(chosen.get('id') or chosen.get('key') or '')

        new_item = dict(diag)
        new_item['snomed_code'] = code_value
        cleaned.append(new_item)

    return {'diagnoses': cleaned}


def main():
    file_path = "./data/doc1.txt"
    pdf_path = "./data/gastroscopy report 02 jun (1).pdf"
    output_pdf_path = "./data/gastroscopy_report.pdf"
    poc = POC()
    # Step 1: upload read document content and create an embedings and store into vector DB
    # text = poc.read_text_document(file_path)
    # poc.create_vectorization(text=text, file_name=file_path)
    # print(f"document chunks created sucessfully for {file_path}...")

    # Step 2: Find Active diagnosis using retrievalQa in langchain
    active_diagnosis = poc.get_active_diagnosis(file_path)
    final_active_diagnosis = json.loads(active_diagnosis)
    print(f"Active diagnosis: \n{final_active_diagnosis}")

    #Step 2.1: Fetch Snowmed codes base on Snowmed CT terms from NHS search API
    snomed_codes = get_snomed_codes_for_active_diagonosis(final_active_diagnosis)
    print(f"Snomed Codes: {snomed_codes}")

    # Step 2.2: Clean diagnosis data by attaching exact SNOMED codes
    cleaned = clean_diagnosis_data(final_active_diagnosis, snomed_codes)
    print(f"Cleaned Active diagnosis: \n{cleaned}")

    # Step 2.3: Highlight terms in PDF and save with SNOMED code annotations
    try:
        highlighted_pdf = highlight_terms_in_pdf(
            pdf_path=pdf_path,
            output_pdf_path=output_pdf_path,
            diagnoses=cleaned['diagnoses'],
            snomed_codes_list=snomed_codes,
            case_sensitive=True
        )
        print(f"PDF with highlights saved to: {highlighted_pdf}")
        print("Open the PDF and click on highlighted terms to see SNOMED codes!")
    except Exception as e:
        print(f"Error highlighting PDF: {e}")
        print("Make sure the PDF file exists and PyMuPDF is installed: pip install pymupdf")

    # Step 3: Find Active procedures using retrievalQa in langchain
    # active_procedures = poc.get_active_procedures(file_path)
    # final_active_procedures = json.loads(active_procedures)
    # print(f"Active Procedures: {final_active_procedures}")

if __name__ == "__main__":
    main()