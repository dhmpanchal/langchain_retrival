import datetime
from pydoc import doc
from bedrock_embedings import BedrockEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.pgvector import PGVector

import os
import constant
import db_helper
import requests
# import boto3
from dotenv import load_dotenv

load_dotenv()
from diagnosis_agent import extract_active_diagnoses
from procedures_agent import extract_active_procedures

class POC:
    def __init__(self):
        self.embeddings = BedrockEmbeddings()
        self.ip = os.getenv("IP",'')

    def read_text_document(self, file_path: str) -> str:
        """
        Reads the content of a text document and returns it as a string.
        Tries utf-8, then latin1, then utf-8 with errors='replace'.
        :param file_path: Path to the text document.
        :return: Content of the document as a string.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='latin1') as file:
                    return file.read()
            except UnicodeDecodeError:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
                    return file.read()

    def create_vectorization(
        self,
        text: str,
        collection_name: str = "medical_notes",
        chunk_size: int = 800,
        chunk_overlap: int = 300,
        bedrock_model_id: str = constant.embedding_model_id,
        aws_region: str = constant.region_name,
        file_name: str = ""
    ):
        """
        Chunk the provided text and store embeddings into PostgreSQL using PGVector.

        Returns (success: bool, message: str)
        """
        print(f"start creating chunks...")
        db_helper.ensure_pgvector_extension()

        # Verify AWS credentials are available (avoid empty embeddings)
        # session_creds = boto3.Session(region_name=aws_region).get_credentials()
        # if session_creds is None:
        #     raise RuntimeError(
        #         "AWS credentials not found. Configure AWS_ACCESS_KEY_ID/SECRET, or an IAM role/profile."
        #     )

        # Health check: ensure embedder returns a non-empty vector
        try:
            probe = self.embeddings.embed_query("healthcheck") or []
        except Exception as exc:
            raise RuntimeError(f"Bedrock embeddings failed: {exc}") from exc
        if not probe or len(probe) == 0:
            raise RuntimeError(
                "Bedrock embeddings returned an empty vector. Verify model access and credentials."
            )

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", "", ". "],
        )

        metadatas = [{
            'filename': file_name,
            'uploaded_at': datetime.datetime.now().isoformat(),
        }]
        documents = text_splitter.create_documents([text], metadatas=metadatas)

        connection_string = PGVector.connection_string_from_db_params(
            driver="psycopg2",
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            database=os.getenv("DB_NAME", "postgres"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", ""),
        )
        
        vectorstore = PGVector(
            embedding_function=self.embeddings,
            connection_string=connection_string,
            collection_name="embeddings"
        )
        vectorstore.add_documents(documents)

        return True, f"Stored {len(documents)} documents under '{file_name}'."
    
    def get_active_diagnosis(self, file_path):
        file_fillter = {"filename": file_path}
        res = extract_active_diagnoses(collection_name="embeddings", filter=file_fillter)
        return res.model_dump_json(indent=2)
    
    def get_active_procedures(self, file_path):
        file_fillter = {"filename": file_path}
        res = extract_active_procedures(collection_name="embeddings", filter=file_fillter)
        return res.model_dump_json(indent=2)
    
    def get_snomed_codes(self, search_term):
        try:
            response_list = []
            url = f"http://{self.ip}/snowstorm/snomed-ct/browser/MAIN/2024-11-20/descriptions?&limit=100&term={search_term}&active=true&conceptActive=true&lang=english&groupByConcept=true"
            response = requests.get(url)

            if response.status_code == 200:
                data = response.json()
                response = data['items']
                response_list = [
                    {
                        "id": item["concept"]["id"],
                        "key":  item["concept"]["conceptId"],
                        "value": item["term"],
                        "description": item["concept"]["fsn"]["term"],
                    }
                    for item in response
                ]
                
                unique_dicts = []
                seen = set()

                for item in response_list:
                    # Create a unique tuple based on the fields to check for duplicates.
                    unique_key = (item["value"], item["key"], item["description"])
                    if unique_key not in seen:
                        unique_dicts.append(item)
                        seen.add(unique_key)
                return unique_dicts
            else:
                return []
        except Exception as e:
            print(f"Error in fetching snomed codes: {e}")
            return []