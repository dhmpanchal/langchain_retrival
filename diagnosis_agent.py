from __future__ import annotations

import os
import json
import re
from typing import List, Optional

from pydantic import BaseModel, Field

from bedrock_embedings import BedrockEmbeddings
from langchain_aws import ChatBedrock
from langchain_community.vectorstores.pgvector import PGVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

import constant
import boto3

from langchain_core.runnables import RunnableConfig

# -----------------------------
# Pydantic schema
# -----------------------------

class DiagnosisItem(BaseModel):
    term: str = Field(..., description="Term from report verbatim")
    snomed_ct_term: str = Field(..., description="SNOMED CT Preferred Term")
    source: str = Field(..., description="Full sentence/line extracted from the document")


class DiagnosisExtraction(BaseModel):
    diagnoses: List[DiagnosisItem] = Field(default_factory=list)


# -----------------------------
# Vector retriever
# -----------------------------

def build_pgvector_retriever(
    collection_name: str = "medical_notes",
    *,
    search_type: str = "mmr",  # or "similarity"
    k: int = 6,
    fetch_k: int = 20,
    lambda_mult: float = 0.3,
    filter: Optional[dict] = None,
):
    """Create a PGVector retriever from existing embeddings in Postgres."""
    embeddings = BedrockEmbeddings()

    connection_string = PGVector.connection_string_from_db_params(
        driver="psycopg2",
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5432")),
        database=os.getenv("DB_NAME", "postgres"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", ""),
    )

    vectorstore = PGVector(
        connection_string=connection_string,
        embedding_function=embeddings,
        collection_name=collection_name,
        use_jsonb=False,
    )

    if search_type == "mmr":
        kwargs = {"k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult}
    else:
        kwargs = {"k": k}
    if filter:
        kwargs["filter"] = filter

    return vectorstore.as_retriever(search_type=search_type, search_kwargs=kwargs)


# -----------------------------
# Bedrock LLM
# -----------------------------

def build_bedrock_llm(
    model_id: str = constant.AWS_BEDROCK_MODEL_ID,
    region_name: str = constant.region_name,
    client=None
):
    client = boto3.client(
        "bedrock-runtime",
        region_name=region_name,
        aws_access_key_id=constant.aws_access_key_id,
        aws_secret_access_key=constant.aws_secret_access_key,
    )
    return ChatBedrock(
        model_id=model_id,
        region_name=region_name,
        model_kwargs={
            "temperature": 0.0,
            "max_tokens": 1000,
        },
        client=client
    )


# -----------------------------
# Prompt and chain
# -----------------------------

PROMPT_TMPL = """
You are a senior clinical coder extracting ACTIVE DIAGNOSES from a medical report.

CRITICAL EXTRACTION RULES — FOLLOW EXACTLY:
1. Only include diagnoses that are:
   - Active and current at the time of the encounter
   - Confirmed diagnoses (not symptoms, signs, or suspected conditions)
   - Not historical, not family history, not planned/future items

2. For EACH diagnosis, output:
   - "term": EXACTLY the same characters, casing, spelling, and word order as they appear inside the "source" string. COPY-PASTE ONLY from source. DO NOT:
     * change word forms (e.g., "Oesophagus" → "Oesophageal")
     * change spelling (e.g., "Hiatal" → "Hiatus")
     * add or remove words
   - "snomed_ct_term": SNOMED CT Preferred Term for the exact "term" (may differ in wording from "term", but "term" itself must remain untouched).
   - "source": The full, unaltered sentence/phrase from the document containing the diagnosis.

3. IMPORTANT:
   - The "term" MUST be an exact substring of "source" with identical spelling and case.
   - If you cannot find a SNOMED CT term, repeat the same text from "term" for "snomed_ct_term".
   - "source" must be copied verbatim from the report, but:
     * Remove leading "Region:" if present
     * Remove image references like "(1 image)"
     * Keep all other characters exactly as in the document
   - Never paraphrase or translate the diagnosis in "term".

4. If a diagnosis is mentioned multiple times, choose the most complete/descriptive occurrence.

5. DO NOT include:
   - Normal findings ("Normal", "unremarkable")
   - Negative statements ("no evidence of", "ruled out")
   - Symptoms/signs only
   - Future/planned procedures

6. Output must follow this JSON schema exactly:
{format_instructions}

7. Output ONLY the JSON object — no explanations.

Document:
{context}

Extract the active diagnoses exactly as per these rules.
"""

# -----------------------------
# Post-processing validation
# -----------------------------
def clean_and_autocorrect_diagnoses(data: DiagnosisExtraction) -> DiagnosisExtraction:
    cleaned = []
    for diag in data.diagnoses:
        src = diag.source
        # Remove unwanted prefixes/noise
        src = re.sub(r"^Region:\s*", "", src)
        src = re.sub(r"\(\d+\s+images?\)", "", src)
        src = re.sub(r"^\d+\s*-\s*", "", src)
        src = src.strip()

        term = diag.term.strip()

        # If exact term not in source, try to auto-correct
        if term not in src:
            # Try to find the longest overlapping phrase between term & source
            term_words = term.split()
            best_match = ""
            for i in range(len(term_words)):
                for j in range(i+1, len(term_words)+1):
                    phrase = " ".join(term_words[i:j])
                    if phrase and phrase in src and len(phrase) > len(best_match):
                        best_match = phrase
            if best_match:
                term = best_match  # auto-correct term
            else:
                # Last resort: use source itself as term
                term = src

        cleaned.append(DiagnosisItem(
            term=term,
            snomed_ct_term=diag.snomed_ct_term.strip(),
            source=src
        ))

    return DiagnosisExtraction(diagnoses=cleaned)

# -----------------------------
# Chain builder
# -----------------------------
def build_chain(
    collection_name: str = "medical_notes",
    *,
    search_type: str = "mmr",
    k: int = 6,
    fetch_k: int = 20,
    lambda_mult: float = 0.3,
    filter: Optional[dict] = None,
):
    retriever = build_pgvector_retriever(
        collection_name=collection_name,
        search_type=search_type,
        k=k,
        fetch_k=fetch_k,
        lambda_mult=lambda_mult,
        filter=filter,
    )

    llm = build_bedrock_llm()

    parser = PydanticOutputParser(pydantic_object=DiagnosisExtraction)
    format_instructions = parser.get_format_instructions()
    llm.bind(format_instructions=format_instructions)

    prompt = ChatPromptTemplate.from_template(PROMPT_TMPL).partial(
        format_instructions=format_instructions
    )

    # Stuff retrieved docs into the prompt as {context}
    doc_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, doc_chain)

    return chain, parser


# -----------------------------
# Public API
# -----------------------------
def extract_json_from_text(text: str) -> Optional[dict]:
    """Extract JSON object from text by finding the first { and last }"""
    try:
        start = text.find('{')
        end = text.rfind('}') + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
    except json.JSONDecodeError:
        pass
    return None

def extract_json_from_text(text: str) -> Optional[dict]:
    """Extract JSON object from text by finding the first { and last }"""
    try:
        start = text.find('{')
        end = text.rfind('}') + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
    except json.JSONDecodeError:
        pass
    return None

def extract_active_diagnoses(
    *,
    collection_name: str = "embeddings",
    search_type: str = "mmr",
    k: int = 6,
    fetch_k: int = 20,
    lambda_mult: float = 0.3,
    filter: Optional[dict] = None,
) -> DiagnosisExtraction:
    """Run RetrievalQA to extract active diagnoses as structured JSON."""
    chain, parser = build_chain(
        collection_name=collection_name,
        search_type=search_type,
        k=k,
        fetch_k=fetch_k,
        lambda_mult=lambda_mult,
        filter=filter,
    )

    out = chain.invoke(
        {"input": "Extract ONLY the active diagnoses as per the rules."},
        config=RunnableConfig(tags=["dh_clinical_coding_poc"])
    )
    answer_text = out.get("answer", "")
    
    # First try to parse with the Pydantic parser
    try:
        print(f"First try to parse with the Pydantic parser")
        parsed = parser.parse(answer_text)
        return parsed
        # return clean_and_autocorrect_diagnoses(parsed)
    except Exception as e:
        print(f"Parser error: {e}")
        # If parser fails, try to extract and parse JSON manually
        try:
            # Try to extract JSON from the text
            json_data = extract_json_from_text(answer_text)
            if json_data:
                if isinstance(json_data, list):
                    return DiagnosisExtraction(diagnoses=[DiagnosisItem(**item) for item in json_data])
                if isinstance(json_data, dict) and "diagnoses" in json_data:
                    return DiagnosisExtraction(**json_data)
            
            # If we get here, try to parse the entire text as JSON
            try:
                data = json.loads(answer_text)
                if isinstance(data, list):
                    return DiagnosisExtraction(diagnoses=[DiagnosisItem(**item) for item in data])
                if isinstance(data, dict) and "diagnoses" in data:
                    return DiagnosisExtraction(**data)
            except json.JSONDecodeError:
                pass
                
            # If all else fails, try to clean the text and parse again
            clean_text = answer_text.strip()
            if clean_text.startswith('```json'):
                clean_text = clean_text[7:].strip('`').strip()
            return parser.parse(clean_text)
            
        except Exception as inner_e:
            print(f"Error processing model output: {inner_e}")
            print("Raw model output:", answer_text)
            raise ValueError(f"Could not parse model output: {inner_e}")



