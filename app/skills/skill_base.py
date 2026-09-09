"""
技能基类模块，定义所有技能的共同接口
"""
import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union, Tuple

logger = logging.getLogger(__name__)

class SkillResult:
    """
    技能执行结果类
    """
    
    def __init__(self, 
                 success: bool = True, 
                 data: Any = None, 
                 error_message: str = None):
        """
        初始化技能执行结果
        
        Args:
            success: 是否成功
            data: 结果数据
            error_message: 错误信息
        """
        self.success = success
        self.data = data
        self.error_message = error_message
        
    def to_dict(self) -> Dict[str, Any]:
        """
        将结果转换为字典
        
        Returns:
            结果字典
        """
        return {
            "success": self.success,
            "data": self.data,
            "error_message": self.error_message
        }
        
    @classmethod
    def success_result(cls, data: Any = None) -> 'SkillResult':
        """
        创建成功结果
        
        Args:
            data: 结果数据
            
        Returns:
            成功结果
        """
        return cls(True, data)
        
    @classmethod
    def error_result(cls, error_message: str, data: Any = None) -> 'SkillResult':
        """
        创建错误结果
        
        Args:
            error_message: 错误信息
            data: 附加数据
            
        Returns:
            错误结果
        """
        return cls(False, data, error_message)

class BaseSkill(ABC):
    """
    技能基类，所有具体技能实现必须继承此类
    """
    
    # 技能默认配置，子类应该覆盖
    DEFAULT_CONFIG = {
        "type": "",         # 技能类型，如detection, recognition等
        "name": "",         # 技能唯一标识符，如coco_detector
        "name_zh": "",      # 技能中文名称
        "description": "",  # 技能描述
        "status": True,     # 技能状态（是否启用）
        "required_models": [],  # 技能所需模型列表
        "params": {}        # 技能具体参数
    }
    
    def __init__(self, config: Union[Dict[str, Any], str] = None):
        """
        初始化技能
        
        Args:
            config: 技能配置字典
        """
        # 支持两种初始化方式：通过配置字典或者直接给名称
        if isinstance(config, dict):
            # 合并默认配置和传入的配置
            self.config = self.get_default_config()
            # 如果传入的配置中有params，则合并params而不是覆盖
            if "params" in config and "params" in self.config:
                self.config["params"].update(config.get("params", {}))
                config_copy = config.copy()
                if "params" in config_copy:
                    del config_copy["params"]
                self.config.update(config_copy)
            else:
                self.config.update(config)
                
            self.name = self.config.get("name", self.__class__.__name__)
            self.name_zh = self.config.get("name_zh", "")
            self.description = self.config.get("description", "")
            self.status = self.config.get("status", True)
            self.skill_id = self.config.get("id", f"{self.name}_{id(self)}")
        else:
            # 使用默认配置
            self.config = self.get_default_config()
            self.name = config if config else self.config.get("name", self.__class__.__name__)
            self.name_zh = self.config.get("name_zh", "")
            self.description = self.config.get("description", "")
            self.status = self.config.get("status", True)
            self.skill_id = f"{self.name}_{id(self)}"
        
        # 初始化跟踪器（用于目标跟踪，支持 sort / bytetrack / botsort）
        params = self.config.get("params", {}) if isinstance(self.config, dict) else {}
        self.tracker = None
        if params.get("enable_default_sort_tracking", True):
            try:
                from app.services.tracker_service import TrackerService
                self.tracker = TrackerService.from_params(params)
            except ImportError:
                self.log("warning", "无法导入跟踪器服务，将跳过目标跟踪")
        
        # 初始化技能
        self._initialize()

        self._enable_timing_log = bool(params.get("enable_timing_log", False))
        self._models_ready = True
        self._models_ready_error: Optional[str] = None
        if self.get_required_models():
            self._models_ready, self._models_ready_error = self._prepare_models_at_startup()
        
    def __str__(self) -> str:
        """返回技能的字符串表示"""
        return f"{self.name}({self.name_zh})(type={self.config.get('type', '')}, status={self.status})"
        
    def _initialize(self) -> None:
        """
        初始化技能，子类可以覆盖此方法进行额外的初始化
        """
        pass
        
    def validate_config(self) -> bool:
        """
        验证配置是否有效
        
        Returns:
            配置有效返回True，否则返回False
        """
        # 如果没有配置，则不需要验证
        if not self.config:
            return True
            
        # 基本验证：确保技能名称匹配
        config_name = self.config.get("name")
        expected_name = self.DEFAULT_CONFIG.get("name")
        if not config_name or (expected_name and config_name != expected_name):
            logger.error(f"技能名称不匹配: 配置={config_name}, 预期={expected_name}")
            return False
            
        return True
        
    @abstractmethod
    def process(self, input_data: Any, context: Any = None, **kwargs) -> Any:
        """
        处理输入数据
        
        Args:
            input_data: 输入数据（图像帧/路径/字典）
            context: 上下文信息（通常为 fence_config）
            **kwargs: 额外参数
            
        Returns:
            处理结果 (SkillResult)
        """
        pass
        
    def is_enabled(self) -> bool:
        """
        判断技能是否启用
        
        Returns:
            技能是否启用
        """
        return self.status
        
    def enable(self) -> None:
        """启用技能"""
        self.status = True
        
    def disable(self) -> None:
        """禁用技能"""
        self.status = False
        
    def log(self, level: str, message: str) -> None:
        """
        记录日志
        
        Args:
            level: 日志级别
            message: 日志消息
        """
        log_method = getattr(logger, level.lower(), logger.info)
        log_method(f"{message}")
    
    def add_tracking_ids(
        self,
        detections: List[Dict],
        image: Any = None,
    ) -> List[Dict]:
        """
        为检测结果添加跟踪ID
        
        Args:
            detections: 检测结果列表
            image: 当前帧图像（BoT-SORT 可选使用）
            
        Returns:
            带跟踪ID的检测结果列表
        """
        if self.tracker:
            return self.tracker.update(detections, image=image)
        return detections

    @staticmethod
    def draw_track_trajectories(
        frame: Any,
        detections: List[Dict],
        track_color_map: Optional[Dict[Any, Tuple[int, int, int]]] = None,
        default_color: Tuple[int, int, int] = (0, 200, 255),
        thickness: int = 2,
    ) -> Any:
        """
        在帧上绘制跟踪轨迹（需 detections 含 track_id 与 track_history）

        Args:
            frame: OpenCV 图像
            detections: 检测结果列表
            track_color_map: track_id -> BGR 颜色
            default_color: 默认轨迹颜色
            thickness: 轨迹线宽
        """
        import cv2
        import numpy as np

        drawn_track_ids = set()
        for detection in detections:
            track_id = detection.get("track_id")
            history = detection.get("track_history") or []
            if track_id is None or track_id in drawn_track_ids or len(history) < 2:
                continue

            drawn_track_ids.add(track_id)
            color = (track_color_map or {}).get(track_id, default_color)
            points = np.array(history, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [points], isClosed=False, color=color, thickness=thickness)

        return frame
    
    @staticmethod
    def _clamp01(v: Any, default: float) -> float:
        try:
            f = float(v)
        except (TypeError, ValueError):
            return default
        return max(0.0, min(1.0, f))

    @staticmethod
    def _norm_fence_polygon(poly: Any) -> List[Tuple[float, float]]:
        """点列表 [{x,y}...] 或 [[x,y]...] -> [(x,y)...]（0-1 归一化坐标）"""
        pts: List[Tuple[float, float]] = []
        for p in poly or []:
            if isinstance(p, dict):
                x, y = p.get("x"), p.get("y")
            elif isinstance(p, (list, tuple)) and len(p) >= 2:
                x, y = p[0], p[1]
            else:
                continue
            if x is None or y is None:
                continue
            try:
                pts.append((float(x), float(y)))
            except (TypeError, ValueError):
                continue
        return pts

    def _fence_regions(self, fence_config: Dict) -> List[Dict[str, Any]]:
        """统一电子围栏为 [{points:[(x,y)..], ratio, invert, name}]（坐标 0-1 归一化）。

        结构：{regions:[{name, points:[{x,y}..], ratio(0-1，默认1), invert}]}
        """
        if not isinstance(fence_config, dict):
            return []
        regions: List[Dict[str, Any]] = []
        for i, r in enumerate(fence_config.get("regions") or []):
            if not isinstance(r, dict):
                continue
            pts = self._norm_fence_polygon(r.get("points"))
            if len(pts) < 3:
                continue
            regions.append({
                "points": pts,
                "ratio": self._clamp01(r.get("ratio"), 1.0),
                "invert": bool(r.get("invert")),
                "name": r.get("name") or f"区域{i + 1}",
            })
        return regions

    def is_fence_config_valid(self, fence_config: Dict) -> bool:
        """检查围栏配置是否有效（至少一个含 ≥3 个点的区域）。"""
        return len(self._fence_regions(fence_config)) > 0

    def _bbox_in_polygon_ratio(self, bbox: List[float], poly_px: List[Tuple[float, float]], grid: int = 12) -> float:
        """目标 bbox 落在多边形内的面积占比（网格采样近似，0-1）。"""
        if len(poly_px) < 3 or len(bbox) < 4:
            return 0.0
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            return 0.0
        inside = 0
        total = grid * grid
        for i in range(grid):
            px = x1 + (i + 0.5) / grid * w
            for j in range(grid):
                py = y1 + (j + 0.5) / grid * h
                if self._point_in_polygon((px, py), poly_px):
                    inside += 1
        return inside / total if total else 0.0

    def is_point_inside_fence(self, point: Tuple[float, float], fence_config: Dict, image_size: Tuple[int, int] = None) -> bool:
        """判断点是否落在任一围栏区域内（像素坐标）。"""
        try:
            regions = self._fence_regions(fence_config)
            if not regions or not image_size:
                return False
            w, h = image_size
            for reg in regions:
                poly_px = [(x * w, y * h) for (x, y) in reg["points"]]
                if self._point_in_polygon(point, poly_px):
                    return True
            return False
        except Exception as e:
            self.log("error", f"判断点是否在围栏内时出错: {str(e)}")
            return False

    # bbox 内相对锚点预设：(rx, ry) 为相对 bbox 宽高的 0-1 系数
    _ANCHOR_PRESETS = {
        "center": (0.5, 0.5),
        "bottom_center": (0.5, 1.0),   # 脚底/接地点
        "top_center": (0.5, 0.0),
        "head": (0.5, 0.33),           # 人头大致位置
        "bottom_left": (0.0, 1.0),
        "bottom_right": (1.0, 1.0),
        "left_center": (0.0, 0.5),
        "right_center": (1.0, 0.5),
    }

    def _anchor_point(self, bbox: List[float], anchor: Any) -> Optional[Tuple[float, float]]:
        """按锚点取 bbox 内的判定点。anchor 可为预设名，或 (rx, ry) 相对系数（0-1）。"""
        if len(bbox) < 4:
            return None
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        if isinstance(anchor, (tuple, list)) and len(anchor) >= 2:
            try:
                rx, ry = float(anchor[0]), float(anchor[1])
            except (TypeError, ValueError):
                rx, ry = 0.5, 0.5
        else:
            rx, ry = self._ANCHOR_PRESETS.get(str(anchor or "center"), (0.5, 0.5))
        return (x1 + (x2 - x1) * rx, y1 + (y2 - y1) * ry)

    def _overrides_detection_point(self) -> bool:
        """子类是否重写了 _get_detection_point（用于 auto 模式判定关键点语义）。"""
        return type(self)._get_detection_point is not BaseSkill._get_detection_point

    def filter_detections_by_fence(
        self,
        detections: List[Dict],
        fence_config: Dict,
        image_size: Tuple[int, int] = None,
        match_mode: str = "auto",
        point_anchor: Any = None,
    ) -> List[Dict]:
        """根据电子围栏过滤检测结果（支持多区域 + 每区域 ratio/invert）。

        判定方式 match_mode：
        - "ratio"：目标 bbox 落在区域内的面积占比 ≥ 该区域 ratio 即命中（invert 取区域外占比）。
        - "point"：用关键点是否落在区域内判定；关键点取 point_anchor（预设名或 (rx,ry)），
          未指定 point_anchor 时回退到 self._get_detection_point()（尊重子类自定义关键点）。
        - "auto"（默认）：显式给了 point_anchor 或子类重写了 _get_detection_point 时按 point，
          否则按 ratio。这样“地面围栏看脚”等既有点判定技能零回归，未定制技能默认走占比。

        命中任一区域即视为“在围栏内”；trigger_mode=outside 时保留未命中的目标。
        """
        regions = self._fence_regions(fence_config)
        if not regions:
            return detections
        if not image_size:
            self.log("warning", "未提供图像尺寸，无法进行围栏判断")
            return detections

        w, h = image_size
        trigger_mode = (fence_config or {}).get("trigger_mode", "inside")
        for reg in regions:
            reg["_poly_px"] = [(x * w, y * h) for (x, y) in reg["points"]]

        mode = match_mode
        if mode == "auto":
            mode = "point" if (point_anchor is not None or self._overrides_detection_point()) else "ratio"

        filtered_results = []
        for detection in detections:
            bbox = detection.get("bbox") or []
            if len(bbox) < 4:
                continue

            hit = False
            if mode == "point":
                pt = self._anchor_point(bbox, point_anchor) if point_anchor is not None \
                    else self._get_detection_point(detection)
                if pt is None:
                    continue
                for reg in regions:
                    inside = self._point_in_polygon(pt, reg["_poly_px"])
                    eff = (not inside) if reg["invert"] else inside
                    if eff:
                        hit = True
                        break
            else:
                for reg in regions:
                    frac = self._bbox_in_polygon_ratio(bbox, reg["_poly_px"])
                    eff = (1.0 - frac) if reg["invert"] else frac
                    thr = reg["ratio"]
                    # 阈值为 0：有任意重叠即命中；否则按占比≥阈值命中
                    if (thr <= 0 and eff > 0) or (thr > 0 and eff >= thr - 1e-9):
                        hit = True
                        break

            keep = (not hit) if trigger_mode == "outside" else hit
            if keep:
                filtered_results.append(detection)

        return filtered_results
    
    def _get_detection_point(self, detection: Dict) -> Optional[Tuple[float, float]]:
        """
        获取检测对象的关键点（用于围栏判断）
        子类可以覆盖此方法来自定义关键点的获取逻辑
        
        Args:
            detection: 检测结果
            
        Returns:
            检测点坐标 (x, y)，如果无法获取则返回None
        """
        # 默认实现：使用检测框的中心点
        bbox = detection.get("bbox", [])
        if len(bbox) >= 4:
            # bbox格式: [x1, y1, x2, y2]
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            return (center_x, center_y)
        return None
    
    def _point_in_polygon(self, point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
        """
        使用射线法判断点是否在多边形内
        
        Args:
            point: 待判断的点 (x, y)
            polygon: 多边形顶点列表 [(x1, y1), (x2, y2), ...]
            
        Returns:
            点是否在多边形内
        """
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
        
    def get_metadata(self) -> Dict[str, Any]:
        """
        获取技能元数据
        
        Returns:
            技能元数据
        """
        return {
            "name": self.name,
            "name_zh": self.name_zh,
            "type": self.config.get("type", ""),
            "description": self.description,
            "status": self.status,
            "id": self.skill_id,
            "required_models": self.get_required_models()
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将技能转换为字典
        
        Returns:
            技能字典
        """
        return {
            **self.get_metadata(),
            "config": self.config
        }
        
    def get_required_models(self) -> List[str]:
        """
        获取技能所需的模型列表
        
        Returns:
            模型名称列表
        """
        # 从配置中获取所需模型列表
        if self.config and "required_models" in self.config:
            return self.config["required_models"]
        return []
        
    def get_model_version(self) -> str:
        """获取 Triton 模型版本号，空字符串表示使用最新版本"""
        params = self.config.get("params", {}) if isinstance(self.config, dict) else {}
        return str(params.get("model_version", "") or "").strip()

    @staticmethod
    def _timing_start() -> float:
        return time.perf_counter()

    @staticmethod
    def _timing_elapsed_ms(start: float) -> float:
        return round((time.perf_counter() - start) * 1000, 2)

    def _attach_process_timing(self, result_data: Dict[str, Any], timing: Dict[str, float]) -> None:
        """将分段耗时写入结果，并在 enable_timing_log 时输出日志。"""
        result_data["timing_ms"] = timing
        if self._enable_timing_log:
            self.log(
                "info",
                "分段耗时(ms): "
                + ", ".join(f"{key}={value}" for key, value in timing.items()),
            )

    def _prepare_models_at_startup(self) -> Tuple[bool, Optional[str]]:
        """
        启动阶段检查 Triton / 模型就绪，并预热 metadata 缓存。
        避免在 process() 每帧重复 RPC。
        """
        ready, err_msg = self.check_model_readiness()
        if not ready:
            return False, err_msg

        from app.services.triton_client import triton_client

        model_version = self.get_model_version()
        for model_name in self.get_required_models():
            if not model_name:
                continue
            if not triton_client.warm_metadata_cache(model_name, model_version):
                ver_label = f" version={model_version}" if model_version else ""
                return False, f"无法获取模型 {model_name}{ver_label} 元数据"

        ver_label = f" v{model_version}" if model_version else ""
        self.log(
            "info",
            f"模型启动检查通过: {', '.join(self.get_required_models())}{ver_label}",
        )
        return True, None

    def ensure_models_ready(self) -> Tuple[bool, Optional[str]]:
        """返回启动阶段模型就绪状态（不再每帧重复检查）。"""
        return self._models_ready, self._models_ready_error

    def check_model_readiness(self) -> Tuple[bool, Optional[str]]:
        """
        检查所需模型和Triton服务器是否就绪
        
        Returns:
            (bool, str): 是否就绪，如果不就绪返回错误信息
        """
        from app.services.triton_client import triton_client
        
        # 检查Triton服务器是否就绪
        if not triton_client.is_server_ready():
            return False, "Triton服务器未就绪"
        
        # 获取所需模型
        required_models = self.get_required_models()
        model_version = self.get_model_version()
        
        # 检查所有模型是否就绪
        for model_name in required_models:
            if model_name and not triton_client.is_model_ready(model_name, model_version):
                ver_label = f" version={model_version}" if model_version else ""
                return False, f"模型 {model_name}{ver_label} 未就绪"
                
        return True, None

    def get_default_config(self) -> Dict[str, Any]:
        """
        获取完整默认配置
        
        Returns:
            Dict[str, Any]: 完整默认配置
        """
        # 复制默认配置
        return self.DEFAULT_CONFIG.copy()

    def analyze_safety(self, detections: List[Dict]) -> Dict[str, Any]:
        """
        分析安全状况（基类默认实现）
        子类应该覆盖此方法实现具体的安全分析逻辑
        
        Args:
            detections: 检测结果列表
            
        Returns:
            安全分析结果字典，包含预警信息
        """
        return {
            "total_detections": len(detections),
            "is_safe": True,             # 默认安全
            "alert_triggered": False,    # 默认不触发预警
            "alert_level": 0,           # 默认预警等级为0
            "message": "基类默认安全分析，建议子类覆盖此方法"
        }

