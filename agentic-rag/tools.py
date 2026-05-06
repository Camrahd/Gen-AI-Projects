import arxiv

def arxiv_tool(query):
    search = arxiv.Search(query=query, max_results=3)
    results = []

    for r in search.results():
        results.append(f"{r.title}: {r.summary[:200]}")

    return "\n".join(results)