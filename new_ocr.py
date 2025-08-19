import fitz
import tempfile
from pdf2image import convert_from_path
import boto3
import re
from math import ceil
import constant
from io import BytesIO
import random

from typing import List, Tuple, Dict, Optional

textract_client = boto3.client('textract',
                                aws_access_key_id= constant.aws_access_key_id,
                                aws_secret_access_key=constant.aws_secret_access_key,
                                region_name=constant.region_name)


def read_pdf_document(pdf_path):
    try:
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
                    words_and_boxes.append({text: [
                        (left, top), (right, top), (right, bottom), (left, bottom)
                    ]})

            all_words_and_boxes.append((page_index, words_and_boxes))
            return all_words_and_boxes
    except Exception as e:
        print(f"Error reading PDF document: {e}")
        return []

def _strip_punct(s: str) -> str:
    # remove leading/trailing punctuation (keep interior characters)
    return re.sub(r'^\W+|\W+$', '', s)

def find_sequence_with_page(
    all_words_and_boxes: List[Tuple[int, List[Dict[str, list]]]],
    input_phrase: str
) -> Optional[Tuple[int, List[Dict[str, list]]]]:
    target_tokens = input_phrase.split()
    target_tokens_norm = [_strip_punct(t) for t in target_tokens]

    for page_index, words_and_boxes in all_words_and_boxes:
        page_words_norm = [_strip_punct(next(iter(d))) for d in words_and_boxes]
        n, m = len(page_words_norm), len(target_tokens_norm)
        if m == 0 or n < m:
            continue
        for i in range(n - m + 1):
            if page_words_norm[i:i+m] == target_tokens_norm:
                return page_index, words_and_boxes[i:i+m]
    return None

if __name__ == "__main__":
    pdf_path = "./data/gastroscopy report 02 jun (1).pdf"
    result = read_pdf_document(pdf_path)
    # print(f"[INFO] OCR coordinates: {result}")
    # input_word = 'Hiatus hernia'
    input_word = 'Oesophageal stricture (Type: Benign)'
    split_input = input_word.split() 
    match = find_sequence_with_page(result, input_word)
    print(f"[INFO] Match: {match}")


