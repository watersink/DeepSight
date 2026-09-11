"""ZLMediaKit HTTP API 客户端"""
import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urlparse

from app.core.config import settings

logger = logging.getLogger(__name__)


class ZLMClientError(Exception):
    """ZLMediaKit API 调用失败"""


def parse_stream_url(in_url: str) -> Optional[Dict[str, str]]:
    """
    从摄像头 in_url 解析 ZLM 流定位信息。

    支持：rtmp://host[:port]/app/stream...
         rtsp://host[:port]/app/stream...
    本地文件等非流地址返回 None。
    """
    raw = (in_url or "").strip()
    if not raw:
        return None
    parsed = urlparse(raw)
    scheme = (parsed.scheme or "").lower()
    if scheme not in {"rtmp", "rtsp", "rtsps", "http", "https"}:
        return None
    path = (parsed.path or "").strip("/")
    if not path:
        return None
    parts = [p for p in path.split("/") if p]
    if parts and parts[-1].endswith(".live.flv"):
        parts[-1] = parts[-1][: -len(".live.flv")]
    if len(parts) < 2:
        return None
    app = parts[0]
    stream = "/".join(parts[1:])
    media_schema = "rtmp" if scheme in {"rtmp", "http", "https"} else "rtsp"
    return {
        "schema": media_schema,
        "vhost": settings.ZLM_DEFAULT_VHOST,
        "app": app,
        "stream": stream,
    }


def _hostname_is_zlm(hostname: Optional[str]) -> bool:
    if not hostname:
        return False
    host = hostname.strip().lower()
    zlm = (settings.ZLM_HOST or "").strip().lower()
    return host in {zlm, "127.0.0.1", "localhost", "::1"}


def is_rtsp_url(url: str) -> bool:
    scheme = (urlparse((url or "").strip()).scheme or "").lower()
    return scheme in {"rtsp", "rtsps"}


def build_zlm_snap_candidates(
    *,
    app: str = "",
    stream: str = "",
    fallback_url: str = "",
) -> List[str]:
    """
    getSnap 候选地址：优先 ZLM RTMP，其次 HTTP-FLV；避免直接截 RTSP。
    """
    out: List[str] = []
    app_s = (app or "").strip().strip("/")
    stream_s = (stream or "").strip().strip("/")
    if app_s and stream_s:
        out.append(settings.build_zlm_pull_url(app_s, stream_s, output_format="rtmp"))
        out.append(settings.build_flv_play_url(app_s, stream_s))

    fb = (fallback_url or "").strip()
    if fb:
        parsed = parse_stream_url(fb)
        if parsed:
            # 任意协议只要能解析出 app/stream，都改写为 RTMP / FLV
            out.append(
                settings.build_zlm_pull_url(
                    parsed["app"], parsed["stream"], output_format="rtmp"
                )
            )
            out.append(settings.build_flv_play_url(parsed["app"], parsed["stream"]))
        # 非 RTSP 的原始地址也可作为最后兜底（如已是 rtmp/http）
        if not is_rtsp_url(fb):
            out.append(fb)

    # 去重保序
    seen = set()
    uniq: List[str] = []
    for u in out:
        if u and u not in seen:
            seen.add(u)
            uniq.append(u)
    return uniq


def resolve_snap_urls_from_raw(url: str) -> List[str]:
    """仅给了 url（无摄像头记录）时，把 RTSP 改写为 ZLM RTMP/FLV。"""
    raw = (url or "").strip()
    if not raw:
        return []
    parsed = parse_stream_url(raw)
    if parsed and (is_rtsp_url(raw) or _hostname_is_zlm(urlparse(raw).hostname)):
        return build_zlm_snap_candidates(
            app=parsed["app"], stream=parsed["stream"], fallback_url=raw
        )
    if is_rtsp_url(raw):
        # 外部摄像头 RTSP 且无法解析 app/stream：无法安全改写，原样返回（调用方应走 proxy）
        return [raw]
    return build_zlm_snap_candidates(fallback_url=raw) or [raw]


class ZLMClient:
    def __init__(
        self,
        host: Optional[str] = None,
        http_port: Optional[int] = None,
        secret: Optional[str] = None,
        timeout: float = 30.0,
    ):
        self.host = host or settings.ZLM_HOST
        self.http_port = http_port or settings.ZLM_HTTP_PORT
        self._secret_override = secret
        self.timeout = timeout
        self.base_url = f"http://{self.host}:{self.http_port}"

    @property
    def secret(self) -> str:
        if self._secret_override is not None:
            return str(self._secret_override)
        return str(settings.ZLM_SECRET or "")

    def _request(self, api: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        query = dict(params or {})
        secret = (self.secret or "").strip()
        if not secret:
            raise ZLMClientError(
                "ZLM_SECRET 为空：请在 .env / 配置中填写与 ZLMediaKit "
                "config.ini [api] secret 一致的密钥"
            )
        # 强制覆盖，避免空 secret 占位导致鉴权失败
        query["secret"] = secret

        url = f"{self.base_url}/index/api/{api}"
        if query:
            url = f"{url}?{urllib.parse.urlencode(query)}"

        req = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            raise ZLMClientError(f"ZLMediaKit HTTP {e.code}: {detail}") from e
        except urllib.error.URLError as e:
            raise ZLMClientError(f"无法连接 ZLMediaKit ({self.base_url}): {e.reason}") from e

        try:
            payload = json.loads(body)
        except json.JSONDecodeError as e:
            raise ZLMClientError(f"ZLMediaKit 返回非 JSON: {body[:200]}") from e

        if payload.get("code") != 0:
            msg = str(payload.get("msg") or payload)
            if "login" in msg.lower() or "登录" in msg:
                raise ZLMClientError(
                    f"{msg}（已携带 secret，长度={len(secret)}；"
                    f"请核对 ZLM_SECRET 是否与 {self.host}:{self.http_port} 上 "
                    f"config.ini 的 [api] secret 完全一致）"
                )
            raise ZLMClientError(msg or f"ZLMediaKit API 失败: {payload}")

        return payload

    @staticmethod
    def media_key(app: str, stream: str, vhost: Optional[str] = None) -> str:
        vh = vhost or settings.ZLM_DEFAULT_VHOST
        return f"{vh}|{app}|{stream}"

    def list_online_media_keys(self) -> Set[str]:
        """
        调用 getMediaList（替代已过期的 isMediaOnline），返回在线流 key 集合。
        key 格式: vhost|app|stream（任一协议在线即算在线）。
        """
        payload = self._request("getMediaList", {})
        data = payload.get("data") or []
        keys: Set[str] = set()
        if not isinstance(data, list):
            return keys
        for item in data:
            if not isinstance(item, dict):
                continue
            app = str(item.get("app") or "").strip()
            stream = str(item.get("stream") or "").strip()
            if not app or not stream:
                continue
            vhost = str(item.get("vhost") or settings.ZLM_DEFAULT_VHOST)
            keys.add(self.media_key(app, stream, vhost))
        return keys

    def is_stream_online(
        self,
        app: str,
        stream: str,
        vhost: Optional[str] = None,
        *,
        online_keys: Optional[Set[str]] = None,
    ) -> bool:
        """判断直播流是否在线（基于 getMediaList）。"""
        vhost = vhost or settings.ZLM_DEFAULT_VHOST
        if online_keys is not None:
            return self.media_key(app, stream, vhost) in online_keys
        for schema in ("rtmp", "rtsp"):
            payload = self._request(
                "getMediaList",
                {"schema": schema, "vhost": vhost, "app": app, "stream": stream},
            )
            data = payload.get("data") or []
            if isinstance(data, list) and len(data) > 0:
                return True
        return False

    def add_stream_proxy(
        self,
        app: str,
        stream: str,
        url: str,
        *,
        vhost: Optional[str] = None,
        retry_count: int = -1,
        rtp_type: int = 0,
        timeout_sec: float = 10.0,
        enable_rtsp: bool = True,
        enable_rtmp: bool = True,
        auto_close: bool = False,
    ) -> str:
        """
        动态添加拉流代理（/index/api/addStreamProxy）。
        返回流唯一标识 key，如 __defaultVhost__/live/cam_001
        """
        app_s = (app or "").strip()
        stream_s = (stream or "").strip()
        src = (url or "").strip()
        if not app_s or not stream_s or not src:
            raise ZLMClientError("addStreamProxy 需要 app、stream、url")
        payload = self._request(
            "addStreamProxy",
            {
                "vhost": vhost or settings.ZLM_DEFAULT_VHOST,
                "app": app_s,
                "stream": stream_s,
                "url": src,
                "retry_count": int(retry_count),
                "rtp_type": int(rtp_type),
                "timeout_sec": float(timeout_sec),
                "enable_rtsp": 1 if enable_rtsp else 0,
                "enable_rtmp": 1 if enable_rtmp else 0,
                "auto_close": 1 if auto_close else 0,
            },
        )
        data = payload.get("data") or {}
        key = str(data.get("key") or "").strip()
        if not key:
            raise ZLMClientError(f"addStreamProxy 未返回 key: {payload}")
        logger.info(
            "addStreamProxy 成功 app=%s stream=%s key=%s url=%s",
            app_s,
            stream_s,
            key,
            src[:120],
        )
        return key

    def del_stream_proxy(self, key: str) -> None:
        """删除拉流代理（/index/api/delStreamProxy）。"""
        k = (key or "").strip()
        if not k:
            return
        self._request("delStreamProxy", {"key": k})
        logger.info("delStreamProxy 成功 key=%s", k)

    def start_record_task(
        self,
        app: str,
        stream: str,
        path: str,
        back_ms: int,
        forward_ms: int,
        vhost: Optional[str] = None,
    ) -> str:
        """
        调用 /index/api/startRecordTask 开始事件录像。

        Returns:
            录像文件在流媒体服务器上的绝对路径。
        """
        vhost = vhost or settings.ZLM_DEFAULT_VHOST
        payload = self._request(
            "startRecordTask",
            {
                "vhost": vhost,
                "app": app,
                "stream": stream,
                "path": path,
                "back_ms": back_ms,
                "forward_ms": forward_ms,
            },
        )
        data = payload.get("data") or {}
        file_path = data.get("path")
        if not file_path:
            raise ZLMClientError("startRecordTask 未返回录像文件路径")
        logger.info(
            "startRecordTask 已触发 app=%s stream=%s path=%s -> %s",
            app,
            stream,
            path,
            file_path,
        )
        return file_path

    def download_bytes(self, url: str, timeout: Optional[float] = None) -> bytes:
        req = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
                return resp.read()
        except urllib.error.URLError as e:
            raise ZLMClientError(f"下载文件失败 ({url}): {e.reason}") from e

    def get_snap(
        self,
        url: str,
        *,
        timeout_sec: int = 10,
        expire_sec: int = 30,
    ) -> bytes:
        """
        调用 ZLM /index/api/getSnap，返回 JPEG 字节。

        注意：该接口直接返回图片二进制，不是 JSON。
        """
        target = (url or "").strip()
        if not target:
            raise ZLMClientError("getSnap 的 url 不能为空")

        secret = (self.secret or "").strip()
        if not secret:
            raise ZLMClientError(
                "ZLM_SECRET 为空：请在 .env / 配置中填写与 ZLMediaKit "
                "config.ini [api] secret 一致的密钥"
            )

        query = {
            "secret": secret,
            "url": target,
            "timeout_sec": int(timeout_sec),
            "expire_sec": int(expire_sec),
        }
        api_url = f"{self.base_url}/index/api/getSnap?{urllib.parse.urlencode(query)}"
        req = urllib.request.Request(api_url, method="GET")
        # FFmpeg 截图可能较慢，超时略大于 timeout_sec
        wait = max(float(self.timeout), float(timeout_sec) + 5.0)
        try:
            with urllib.request.urlopen(req, timeout=wait) as resp:
                content_type = (resp.headers.get("Content-Type") or "").lower()
                data = resp.read()
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            raise ZLMClientError(f"getSnap HTTP {e.code}: {detail}") from e
        except urllib.error.URLError as e:
            raise ZLMClientError(f"无法连接 ZLMediaKit ({self.base_url}): {e.reason}") from e

        if not data:
            raise ZLMClientError("getSnap 返回空内容")

        # 失败时 ZLM 有时返回 JSON 错误
        if "json" in content_type or data[:1] in (b"{", b"["):
            try:
                payload = json.loads(data.decode("utf-8", errors="replace"))
            except Exception:
                payload = None
            if isinstance(payload, dict) and payload.get("code") not in (None, 0):
                raise ZLMClientError(str(payload.get("msg") or payload))
            raise ZLMClientError(f"getSnap 未返回图片: {data[:200]!r}")

        if len(data) < 100 or data[:2] != b"\xff\xd8":
            # 非标准 JPEG 头也尝试返回，但给出日志
            logger.warning(
                "getSnap 响应可能不是 JPEG content_type=%s head=%s",
                content_type,
                data[:16],
            )
        return data

    def get_snap_prefer_zlm(
        self,
        urls: List[str],
        *,
        timeout_sec: int = 10,
        expire_sec: int = 30,
    ) -> bytes:
        """
        按候选列表依次 getSnap（通常 RTMP → HTTP-FLV），全部失败再抛错。
        """
        candidates = [u.strip() for u in (urls or []) if (u or "").strip()]
        if not candidates:
            raise ZLMClientError("getSnap 无可用流地址")

        errors: List[str] = []
        for idx, target in enumerate(candidates):
            try:
                logger.info(
                    "getSnap 尝试 [%d/%d] url=%s",
                    idx + 1,
                    len(candidates),
                    target[:160],
                )
                return self.get_snap(
                    target, timeout_sec=timeout_sec, expire_sec=expire_sec
                )
            except ZLMClientError as e:
                errors.append(f"{target}: {e}")
                logger.warning("getSnap 失败 url=%s err=%s", target[:160], e)

        raise ZLMClientError(
            "getSnap 全部候选失败（已优先 RTMP/HTTP-FLV，避免直截 RTSP）。"
            + " | ".join(errors[:3])
        )


zlm_client = ZLMClient()
