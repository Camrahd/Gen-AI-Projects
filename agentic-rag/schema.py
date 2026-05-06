from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class Document:
    text: str
    metadata: Dict[str, Any]