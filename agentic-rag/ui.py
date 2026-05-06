import gradio as gr
from ingestion import fetch_arxiv_papers
from chunking import chunk_text
from retriever import HybridRetriever
from schema import Document
from agent import Agent

print("Loading papers...")
docs = fetch_arxiv_papers(50)

chunked_docs = []
for d in docs:
    for c in chunk_text(d.text):
        chunked_docs.append(Document(text=c, metadata=d.metadata))

retriever = HybridRetriever(chunked_docs)
agent = Agent(retriever)

def chat_fn(message, history):
    answer, logs = agent.run(message)
    history = history or []
    history.append((message, answer))
    return history, "", logs

with gr.Blocks() as demo:
    gr.Markdown("# 🤖 Agentic RAG System")

    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Ask something", placeholder="Type your question here...")
    clear = gr.Button("Clear")
    debug = gr.Markdown(label="🧠 Agent Trace")

    msg.submit(chat_fn, [msg, chatbot], [chatbot, msg, debug])
    clear.click(lambda: ([], "", ""), None, [chatbot, msg, debug])

demo.launch()
