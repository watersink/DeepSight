"""一次性脚本：从 ultralytics-main 同步跟踪器源码并修正 import 路径"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "ultralytics-main" / "ultralytics" / "trackers"
DST = ROOT / "app" / "services" / "trackers"

FILES = [
    "byte_tracker.py",
    "bot_sort.py",
    "basetrack.py",
    "oc_sort.py",
    "fast_tracker.py",
    "deep_oc_sort.py",
    "track_tracker.py",
]
UTIL_FILES = ["stracks.py", "reid.py"]


def patch(content: str, in_utils: bool = False) -> str:
    content = content.replace("from __future__ import annotations\n\n", "")
    content = re.sub(
        r"from \.\.utils import LOGGER\n",
        "import logging\n\nLOGGER = logging.getLogger(__name__)\n",
        content,
    )
    content = content.replace("from ultralytics.utils import LOGGER\n", "import logging\n\nLOGGER = logging.getLogger(__name__)\n")
    content = content.replace("from ultralytics.utils.metrics import bbox_ioa\n", "from .utils.metrics import bbox_ioa\n")
    content = content.replace("from ..utils.metrics import bbox_ioa\n", "from .utils.metrics import bbox_ioa\n")
    content = content.replace("from ..utils.ops import xywh2ltwh", "from .utils.ops import xywh2ltwh")
    content = content.replace("from ..utils.ops import xyxy2ltwh", "from .utils.ops import xyxy2ltwh")
    content = content.replace("from ultralytics.utils.ops import xywh2xyxy\n", "")
    content = content.replace("from ultralytics.utils.plotting import save_one_box\n", "")
    content = content.replace("from ultralytics.nn.autobackend import AutoBackend\n", "")
    content = content.replace("import torch\n\n", "")
    content = content.replace("import torch\n", "")
    if in_utils:
        content = content.replace("from ..basetrack import TrackState", "from ...basetrack import TrackState")
        content = content.replace("from . import matching", "from . import matching")
    return content


def main():
    for name in FILES:
        src = SRC / name
        if not src.exists():
            print(f"skip missing {src}")
            continue
        text = patch(src.read_text(encoding="utf-8"))
        (DST / name).write_text(text, encoding="utf-8")
        print(f"synced {name}")

    for name in UTIL_FILES:
        src = SRC / "utils" / name
        if not src.exists():
            continue
        if name == "reid.py":
            continue  # keep simplified local reid.py
        text = patch(src.read_text(encoding="utf-8"), in_utils=True)
        (DST / "utils" / name).write_text(text, encoding="utf-8")
        print(f"synced utils/{name}")


if __name__ == "__main__":
    main()
