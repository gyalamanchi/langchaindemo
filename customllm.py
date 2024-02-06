import requests
import os
from typing import Any, List, Mapping, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM

class CustomLLM(LLM):
    #n: int = 1024

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        API_TOKEN = os.environ["API_TOKEN"] #Set a API_TOKEN environment variable before running
        API_URL = "https://z8dvl7fzhxxcybd8.eu-west-1.aws.endpoints.huggingface.cloud/" #Add a URL for a model of your choosing
        headers = {"Authorization": f"Bearer {API_TOKEN}"}

        payload = {
            "inputs": prompt,
            "parameters": { #Try and experiment with the parameters
                "max_new_tokens": 1024,
                "temperature": 0.6,
                "top_p": 0.9,
                "do_sample": False,
                "return_full_text": False
            }
        }
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()[0]['generated_text']
        #return prompt[: self.n]

    #@property
    #def _identifying_params(self) -> Mapping[str, Any]:
    #    """Get the identifying parameters."""
    #    return {"n": self.n}