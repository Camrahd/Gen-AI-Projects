import arxiv
from schema import Document

def fetch_arxiv_papers(max_results=50):
    search = arxiv.Search(
        query="cat:cs.AI",
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    docs = []
    for result in search.results():
        docs.append(Document(
            text=result.summary,
            metadata={
                "title": result.title,
                "authors": [a.name for a in result.authors],
                "published": str(result.published),
                "arxiv_id": result.entry_id
            }
        ))
    return docs