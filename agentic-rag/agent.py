from openai import OpenAI
from query_rewrite import rewrite_query
from tools import arxiv_tool
from memory import SemanticMemory

client = OpenAI()

class Agent:
    def __init__(self, retriever):
        self.retriever = retriever
        self.memory = SemanticMemory()
        self.history = []

    def decide(self, query, history):
        history_text = "\n".join(
            f"User: {q}\nAssistant: {a}" for q, a in history[-3:]
        ) if history else "No prior conversation."

        prompt = f"""You are a routing agent. Based on the conversation and query, decide which action to take.

Conversation:
{history_text}

Current Query: {query}

Reply with exactly ONE of these words, then a brief reason:
- RETRIEVE: search the local knowledge base of AI papers
- TOOL: fetch live/new papers directly from arXiv
- ASK: the query is too ambiguous to answer
- REFUSE: the query is off-topic or inappropriate
- ANSWER: you can answer directly from conversation history

Decision:"""

        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return res.choices[0].message.content

    def run(self, query):
        logs = []

        decision = self.decide(query, self.history)
        logs.append(f"[Decision] {decision}")

        context = []

        if "RETRIEVE" in decision:
            queries = rewrite_query(query)
            logs.append(f"[Rewrite] {queries}")
            for q in queries:
                docs = self.retriever.retrieve(q)
                logs.append(f"[Retrieve] {len(docs)} docs for: '{q}'")
                context.extend(docs)

        elif "TOOL" in decision:
            tool_res = arxiv_tool(query)
            logs.append("[Tool] Used arXiv live search")
            context.append(tool_res)

        elif "ASK" in decision:
            return "Could you please clarify your question?", "\n".join(logs)

        elif "REFUSE" in decision:
            return "I can only answer questions related to AI research papers.", "\n".join(logs)

        # Search semantic memory for relevant past interactions
        memory_hits = self.memory.search(query)
        if memory_hits:
            logs.append(f"[Memory] Found {len(memory_hits)} relevant past interactions")

        context_text = "\n---\n".join([
            c.text if hasattr(c, "text") else str(c)
            for c in context[:5]
        ])
        memory_text = "\n".join(memory_hits) if memory_hits else ""

        sections = []
        if context_text:
            sections.append(f"Context from knowledge base:\n{context_text}")
        if memory_text:
            sections.append(f"Relevant past conversation:\n{memory_text}")

        prompt = f"""You are a helpful AI research assistant. Answer the question using the provided context.

{chr(10).join(sections)}

Question: {query}

Answer:"""

        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        answer = res.choices[0].message.content

        self.memory.add(query + " " + answer)
        self.history.append((query, answer))

        return answer, "\n".join(logs)
