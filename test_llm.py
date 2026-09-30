"""test_llm.py - smoke test: can we reach both Azure OpenAI deployments?"""

import time

from tools.llm import CHAT_DEPLOYMENT, EMBEDDING_DEPLOYMENT, chat, embed

print(f"1) Chat model: {CHAT_DEPLOYMENT}")
start = time.time()
response = chat([
    {"role": "system", "content": "You are a maintenance assistant for a bottling plant. Answer in one sentence."},
    {"role": "user", "content": "Product temperature at the filler is rising above 5 C. What is the most likely cause?"},
])
print("   Answer :", response.choices[0].message.content)
print(f"   Tokens : {response.usage.prompt_tokens} in, {response.usage.completion_tokens} out")
print(f"   Latency: {time.time() - start:.2f}s")

print(f"\n2) Embedding model: {EMBEDDING_DEPLOYMENT}")
vectors = embed(["chiller failure raises product temperature", "bearing wear increases vibration"])
print(f"   Got {len(vectors)} vectors, each with {len(vectors[0])} numbers")
print(f"   First 5 numbers of vector 1: {[round(x, 4) for x in vectors[0][:5]]}")

print("\nAzure OpenAI connection works.")
