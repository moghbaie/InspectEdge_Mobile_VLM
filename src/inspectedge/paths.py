"""Project root resolution. On Colab set INSPECTEDGE_ROOT=/content/drive/MyDrive/InspectEdge."""
import os
from pathlib import Path

ROOT = Path(os.environ.get("INSPECTEDGE_ROOT", Path(__file__).resolve().parents[2]))
