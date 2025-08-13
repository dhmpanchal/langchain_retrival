import json
from typing import List
from langchain.embeddings.base import Embeddings

import constant
import boto3

class BedrockEmbeddings(Embeddings):
    def __init__(self, model_id: str = "amazon.titan-embed-text-v2:0", region: str = "eu-west-2"):
        self.model_id = model_id
        self.client = boto3.client("bedrock-runtime", region_name=region,aws_access_key_id=constant.aws_access_key_id, aws_secret_access_key=constant.aws_secret_access_key)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)

    def _embed(self, text: str) -> List[float]:
        try:
            response = self.client.invoke_model(
                modelId=self.model_id,
                contentType=constant.application_json,
                accept=constant.application_json,
                body=json.dumps({"inputText": text,"dimensions": 512,"normalize": True})
            )
            result = json.loads(response["body"].read())
            return result["embedding"]
        except Exception as e:
            print(f"Bedrock embedding error: {e}")
            return []