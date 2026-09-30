"""
llm.py - one shared connection to Azure OpenAI for every agent.

Reads settings from environment variables (.env locally, App Settings in Azure),
so no secret is ever hard-coded.
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # loads .env into environment variables

ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/")
API_KEY = os.environ["AZURE_OPENAI_API_KEY"]
CHAT_DEPLOYMENT = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4.1-mini")
EMBEDDING_DEPLOYMENT = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")

# Azure OpenAI "v1" API: the standard OpenAI SDK pointed at your Azure endpoint
client = OpenAI(base_url=f"{ENDPOINT}/openai/v1/", api_key=API_KEY)


def chat(messages: list[dict], temperature: float = 0.0, **kwargs):
    """Send a chat request to the deployed chat model."""
    return client.chat.completions.create(
        model=CHAT_DEPLOYMENT,  # on Azure, "model" = your deployment name
        messages=messages,
        temperature=temperature,
        **kwargs,
    )


def embed(texts: list[str]) -> list[list[float]]:
    """Turn a list of texts into embedding vectors."""
    response = client.embeddings.create(model=EMBEDDING_DEPLOYMENT, input=texts)
    return [item.embedding for item in response.data]
