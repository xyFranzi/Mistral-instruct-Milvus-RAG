import sys
sys.path.append("/home/franzi/mypython/Mistral-instruct-Milvus-RAG")
from config import OPENAI_API_KEY
from openai import OpenAI

client = OpenAI(api_key=OPENAI_API_KEY)

response = client.responses.create(
    model="gpt-4.1",
    tools=[{"type": "web_search_preview"}],
    input="Tell me about the women's EURO 2024 current results and standings."
)

print(response.output_text)