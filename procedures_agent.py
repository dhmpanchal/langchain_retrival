from __future__ import annotations

import os
import json
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

class ProceduresItem(BaseModel):
    procedure: str = Field(..., description="Term from report verbatim")
    snomed_ct_term: str = Field(..., description="SNOMED CT Preferred Term")
    source: str = Field(..., description="Full sentence/line extracted from the document")


class ProcedureExtraction(BaseModel):
    procedures: List[ProceduresItem] = Field(default_factory=list)


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
    model_id: str = constant.AWS_BEDROCK_MODEL_LLM,
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

PROMPT_TMPL = (
    "You are a senior clinical coder. Your job is to read clinical text and extract only PROCEDURES "
    "performed, initiated, or actively planned for the patient during the current encounter.\n\n"
    "Strict rules:\n"
    "Output only procedures that are documented as having been carried out, started, or clearly scheduled as part of the current visit.\n"
    "Use the exact wording as it appears in the document. Do NOT rephrase, expand, or normalize (e.g., keep “Monitor her liver function” if written that way).\n"
    "INCLUDE:\n"
    "  • therapeutic procedures (e.g., physiotherapy, joint protection strategies)\n"
    "  • diagnostic procedures/tests (e.g., ESR, CRP test, X-ray)\n"
    "  • surgical procedures (if mentioned)\n"
    "  • monitoring procedures (e.g., Monitor her liver function, Monitor blood counts)\n"
    "  • administration of treatment (e.g., IV infusion, injection, vaccination)\n"
    "EXCLUDE:\n"
    "  • diagnoses, conditions, symptoms, and signs\n"
    "  • negations (e.g., “no procedure performed”)\n"
    "  • past history procedures unless being repeated now\n"
    "  • vague advice without clinical action (e.g., 'encouraged to eat healthy')\n"
    "  • hypothetical or potential future options not yet confirmed (e.g., 'may consider surgery')\n"
    "If uncertain whether an item is a procedure, exclude it.\n"
    "If no procedures are present, return an empty list.\n\n"
    "Return format: a JSON object with a top-level key 'procedures' containing an array of objects.\n"
    "Schema (follow strictly):\n{format_instructions}\n\n"
    "Document:\n{context}\n\n"
    "Query: Extract ONLY the procedures as per the rules. Return ONLY a JSON object with 'procedures' as specified."
)


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

    parser = PydanticOutputParser(pydantic_object=ProcedureExtraction)
    format_instructions = parser.get_format_instructions()

    prompt = ChatPromptTemplate.from_template(PROMPT_TMPL).partial(
        format_instructions=format_instructions
    )

    # Stuff retrieved docs into the prompt as {context}
    doc_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, doc_chain)

    return chain, parser


# -----------------------------
# JSON extraction helper
# -----------------------------

def _extract_json_block(text: str) -> str:
    """Extract the first valid JSON object/array from text (handles preamble)."""
    if not text:
        return text
    # Strip common code fences
    txt = text.strip()
    if txt.startswith("```"):
        # remove leading and trailing fences if present
        parts = txt.split("```")
        # pick the largest block between fences
        txt = max(parts, key=len).strip()
    # Find first JSON starting point
    start = None
    for i, ch in enumerate(txt):
        if ch in "[{":
            start = i
            break
    if start is None:
        return txt
    # Balance braces/brackets
    stack = []
    end = None
    for i in range(start, len(txt)):
        c = txt[i]
        if c in "[{":
            stack.append(c)
        elif c in "]}":
            if not stack:
                continue
            opening = stack.pop()
            if (opening == "[" and c != "]") or (opening == "{" and c != "}"):
                continue
            if not stack:
                end = i + 1
                break
    return txt[start:end] if end else txt[start:]


# -----------------------------
# Public API
# -----------------------------

def extract_active_procedures(
    *,
    collection_name: str = "medical_notes",
    search_type: str = "mmr",
    k: int = 6,
    fetch_k: int = 20,
    lambda_mult: float = 0.3,
    filter: Optional[dict] = None,
) -> ProcedureExtraction:
    """Run RetrievalQA to extract active diagnoses as structured JSON."""
    chain, parser = build_chain(
        collection_name=collection_name,
        search_type=search_type,
        k=k,
        fetch_k=fetch_k,
        lambda_mult=lambda_mult,
        filter=filter,
    )

    # The retrieval chain returns {"answer": str, "context": List[Document]}
    out = chain.invoke(
        {"input": "Extract ONLY the procedures as per the rules. Return a JSON object with 'procedures' as specified."},
        config=RunnableConfig(tags=["dh_clinical_coding_poc"])
    )
    answer_text = out.get("answer", "").strip()

    # Parse to Pydantic model; fallback to extracting JSON block then wrapping if needed
    try:
        result: ProcedureExtraction = parser.parse(answer_text)
        return result
    except Exception:
        cleaned = _extract_json_block(answer_text)
        try:
            data = json.loads(cleaned)
        except Exception as e:
            raise ValueError(f"Model output is not valid JSON: {e}: {answer_text}") from e
        if isinstance(data, list):
            return ProcedureExtraction(procedures=[ProceduresItem(**item) for item in data])
        if isinstance(data, dict) and "procedures" in data:
            return ProcedureExtraction(**data)
        raise ValueError(f"Unexpected JSON shape. Expected list or object with 'procedures': {data}")



