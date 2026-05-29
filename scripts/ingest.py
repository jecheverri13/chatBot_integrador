import sys
from pathlib import Path

# Permite `python scripts/ingest.py` desde cualquier cwd: la raíz del repo debe estar en sys.path.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.rag import RagService


if __name__ == "__main__":
    service = RagService()
    result = service.build_index()
    print(result)
