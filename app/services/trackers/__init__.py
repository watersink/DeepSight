# Ultralytics YOLO 🚀, AGPL-3.0 license
"""跟踪器按需导入。包初始化不加载 ByteTrack 等依赖，避免缺 scipy/lap 时 SORT 也无法使用。"""

__all__ = (
    "Sort",
    "BYTETracker",
    "BOTSORT",
    "OCSORT",
    "DeepOCSORT",
    "FASTTracker",
    "TRACKTRACK",
)

_LAZY_IMPORTS = {
    "Sort": (".sort", "Sort"),
    "BYTETracker": (".byte_tracker", "BYTETracker"),
    "BOTSORT": (".bot_sort", "BOTSORT"),
    "OCSORT": (".oc_sort", "OCSORT"),
    "DeepOCSORT": (".deep_oc_sort", "DeepOCSORT"),
    "FASTTracker": (".fast_tracker", "FASTTracker"),
    "TRACKTRACK": (".track_tracker", "TRACKTRACK"),
}


def __getattr__(name):
    target = _LAZY_IMPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(target[0], __name__)
    value = getattr(module, target[1])
    globals()[name] = value
    return value
