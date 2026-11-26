import sys
from pathlib import Path

# Add the repo root to sys.path so pytest can find 'pipeline' and 'tests'
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))