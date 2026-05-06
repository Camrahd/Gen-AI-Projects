import re
from openai import OpenAI

client = OpenAI()

def rewrite_query(query):
    prompt = f"Generate 3 alternative search queries for: {query}\nReturn just the queries, one per line, no numbering or bullets."

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    lines = res.choices[0].message.content.split("\n")
    queries = [re.sub(r'^\d+[\.\)]\s*', '', line).strip() for line in lines]
    return [q for q in queries if q]
