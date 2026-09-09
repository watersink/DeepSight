"""视频流处理共享类型定义"""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import numpy as np

# 告警回调: (result_data, raw_frame, scene_id) -> None
AlertCallback = Callable[[Dict[str, Any], np.ndarray, str], None]

# 是否触发告警的判断函数: (result_data) -> bool
AlertPredicate = Callable[[Dict[str, Any]], bool]


@dataclass
class StreamConfig:
    """视频流管道配置"""

    source_url: str
    output_url: str
    fps: float = 15.0
    alarm_interval: float = 1.0
    scene_id: str = ""
    queue_size: int = 4
    output_format: str = "rtmp"  # rtmp | rtsp
    fence_config: Optional[Dict[str, Any]] = None
    use_hardware_encoding: bool = True
    bitrate: str = "2M"
    buffer_size: str = "2M"
    reconnect_delay: float = 2.0
    extra: Dict[str, Any] = field(default_factory=dict)


def default_alert_predicate(data: Dict[str, Any]) -> bool:
    """默认触发条件：过线计数变化、违规、或画面人数变化。"""
    return bool(
        data.get("has_enter_count_change")
        or data.get("has_bypass_violation")
        or data.get("has_count_exit_violation")
        or data.get("has_person_count_change")
    )
