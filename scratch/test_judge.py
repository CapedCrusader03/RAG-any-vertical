import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemma-4-31b-it")

client = OpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=GEMINI_API_KEY
)

context = "This is a placeholder policy context. Texas Homeowners Policy covers water damage up to $10,000."
refusal_answer = "Refusal: The generated answer contains unverifiable citations ['No citations provided in the answer.'] or is missing citations. Please revise query."

judge_prompt = f"""You are an expert compliance auditor. Determine if the Candidate Answer is fully faithful to the provided Context and contains NO hallucinated claims not supported by the context.

Context:
{context}

Candidate Answer:
{refusal_answer}

Rate the faithfulness on a scale from 0 to 5, where:
0 = Completely hallucinated or contradicts context.
5 = 100% faithful, every claim is fully supported by the context.

Respond with ONLY the integer score number (e.g. 5 or 3).
Score:"""

try:
    response = client.chat.completions.create(
        model=GEMINI_MODEL,
        messages=[{"role": "user", "content": judge_prompt}],
        temperature=0.0,
        max_tokens=512
    )
    print("Raw LLM Judge Output:")
    print(response.choices[0].message.content)
except Exception as e:
    print(f"Error: {e}")
