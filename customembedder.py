import os
from typing import List
import requests

from chromadb.api.types import (
    Documents,
    EmbeddingFunction,
    Embeddings
)

class CustomEmbedder(EmbeddingFunction):
    def __init__(self) -> None:
        self.API_TOKEN = os.environ["API_TOKEN"]
        self.API_URL = "https://c9ejquzh6yum3xqf.us-east-1.aws.endpoints.huggingface.cloud/"

    def __call__(self, input: Documents) -> Embeddings:
        rest_client = requests.Session()
        response = rest_client.post(
            self.API_URL, json={"inputs": input}, headers={"Authorization": f"Bearer {self.API_TOKEN}"}
        ).json()
        #print(response)
        return response
    
    #def embed_documents(self,texts) -> List[List[float]]:
        #print("I AM CALLING EMBEDDING FUNCTION")
        #return self(texts)
        
        #Can you convert x into floats which is currently a string into a list of floats
    #    print(self.__call__(texts))
    #    return (self.__call__(texts))
    
    #def embed_query(self, text: str) -> List[float]:
    #    print("I AM CALLING EMBEDDING QUERY")
    #    return self.embed_documents([text])[0]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.__call__(texts)