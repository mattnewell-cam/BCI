import sys
from pathlib import Path


def add_repo_root():
    """Ensure repo root is on sys.path for imports like `utils`."""
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "AGENTS.md").exists() or (parent / ".git").exists():
            root = parent
            break
    else:
        root = here.parent.parent

    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root
