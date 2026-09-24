"""证据图片 Base64 解码 + 上传文件服务（MinIO）共用逻辑。

规则（煤安平台 6.3 / 7.3）：
- ``imagesBase64`` 为空时跳过图片上传，``localFiles`` 为空数组；
- 单张支持纯 Base64 或 ``data:image/xxx;base64,`` 前缀格式；
- 解码后逐张上传文件服务，返回 URL 数组写入 ``localFiles``；
- 单张上传失败只记日志，不影响该条记录入库。
"""
from __future__ import annotations

import base64
import binascii
import logging
import re
import uuid
from datetime import datetime
from typing import Any, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# 图片后缀 / MIME：按 magic number 匹配
_IMAGE_SIGNATURES: Tuple[Tuple[bytes, str, str], ...] = (
    (b"\xff\xd8\xff", ".jpg", "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", ".png", "image/png"),
    (b"GIF87a", ".gif", "image/gif"),
    (b"GIF89a", ".gif", "image/gif"),
)
_DEFAULT_IMAGE_EXT = ".jpg"
_DEFAULT_IMAGE_MIME = "image/jpeg"

# 单张图片大小上限（20MB），防止异常 Base64 撑爆内存
MAX_IMAGE_BYTES = 20 * 1024 * 1024

_DATA_URL_RE = re.compile(
    r"^data:(?P<mime>image/[A-Za-z0-9.+-]+)?(?P<params>;[^,]*)?,(?P<data>.*)$",
    re.IGNORECASE | re.DOTALL,
)
_SAFE_CODE_RE = re.compile(r"[^0-9A-Za-z_.-]+")


def strip_data_url_prefix(raw: str) -> Tuple[str, Optional[str]]:
    """去掉 ``data:image/xxx;base64,`` 前缀，返回 ``(纯Base64, mime)``。"""
    text = str(raw or "").strip()
    if not text.lower().startswith("data:"):
        return text, None
    match = _DATA_URL_RE.match(text)
    if not match:
        return text, None
    mime = str(match.group("mime") or "").strip().lower() or None
    return str(match.group("data") or "").strip(), mime


def guess_image_type(data: bytes, mime: Optional[str] = None) -> Tuple[str, str]:
    """按 magic number 判定后缀与 MIME，识别不出时回退 JPEG。"""
    for signature, ext, mime_type in _IMAGE_SIGNATURES:
        if data.startswith(signature):
            return ext, mime_type
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp", "image/webp"
    if data[:2] in (b"II", b"MM"):
        return ".tif", "image/tiff"
    if data[:2] == b"BM":
        return ".bmp", "image/bmp"
    if mime:
        subtype = mime.split("/", 1)[-1]
        if subtype in {"jpeg", "jpg"}:
            return ".jpg", "image/jpeg"
        if subtype:
            return f".{subtype}", mime
    logger.warning("证据图片：无法识别图片格式，按 JPEG 上传")
    return _DEFAULT_IMAGE_EXT, _DEFAULT_IMAGE_MIME


def decode_base64_image(raw: Any) -> Optional[Tuple[bytes, str, str]]:
    """将单张图片 Base64 解码为 ``(bytes, ext, mime)``；失败返回 None。"""
    text, mime = strip_data_url_prefix(raw)
    if not text:
        return None
    try:
        compact = re.sub(r"\s+", "", text)
        padding = "=" * (-len(compact) % 4)
        data = base64.b64decode(compact + padding, validate=False)
    except (binascii.Error, ValueError):
        logger.warning("证据图片：Base64 解码失败，已跳过该张")
        return None

    if not data:
        logger.warning("证据图片：解码后为空，已跳过该张")
        return None
    if len(data) > MAX_IMAGE_BYTES:
        logger.warning(
            "证据图片：单张过大 size=%s 超过上限 %s，已跳过该张",
            len(data),
            MAX_IMAGE_BYTES,
        )
        return None

    ext, mime_type = guess_image_type(data, mime)
    return data, ext, mime_type


def build_object_name(
    prefix: str,
    camera_code: str,
    data_time: str,
    ext: str,
) -> str:
    """生成对象名：``{prefix}/{cameraCode}/{yyyyMMdd}/{HHMMSSffffff}-{uuid8}{ext}``。"""
    safe_prefix = str(prefix or "").strip().strip("/") or "coal/misc"
    safe_camera = _SAFE_CODE_RE.sub("_", str(camera_code or "").strip()) or "unknown"
    text = str(data_time or "").strip()
    day = ""
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            day = datetime.strptime(text, fmt).strftime("%Y%m%d")
            break
        except ValueError:
            continue
    if not day:
        day = datetime.now().strftime("%Y%m%d")
    stamp = datetime.now().strftime("%H%M%S%f")
    return f"{safe_prefix}/{safe_camera}/{day}/{stamp}-{uuid.uuid4().hex[:8]}{ext}"


def upload_image(
    raw: Any,
    *,
    prefix: str,
    camera_code: str,
    data_time: str,
) -> Optional[str]:
    """解码并上传单张证据图片，返回可访问 URL；失败只记日志并返回 None。"""
    decoded = decode_base64_image(raw)
    if decoded is None:
        return None
    data, ext, content_type = decoded

    object_name = build_object_name(prefix, camera_code, data_time, ext)

    # 延迟导入：未安装 minio SDK 时也不影响模块导入
    try:
        from app.services.minio_client import MinioClientError, minio_storage
    except ImportError as exc:
        logger.error(
            "证据图片：文件服务不可用（minio SDK 未安装）object=%s err=%s",
            object_name,
            exc,
        )
        return None

    try:
        url = minio_storage.upload_bytes(data, object_name, content_type)
    except MinioClientError as exc:
        logger.error(
            "证据图片：上传失败 cameraCode=%s object=%s err=%s",
            camera_code,
            object_name,
            exc,
        )
        return None
    except Exception:
        logger.exception(
            "证据图片：上传异常 cameraCode=%s object=%s",
            camera_code,
            object_name,
        )
        return None

    logger.info(
        "证据图片已上传 cameraCode=%s object=%s size=%s url=%s",
        camera_code,
        object_name,
        len(data),
        url,
    )
    return url


def build_local_files(
    images_base64: Optional[Sequence[str]],
    *,
    prefix: str,
    camera_code: str,
    data_time: str,
) -> List[str]:
    """逐张上传图片，返回文件服务 URL 数组（顺序与入参一致，失败项丢弃）。"""
    urls: List[str] = []
    for raw in images_base64 or []:
        url = upload_image(
            raw, prefix=prefix, camera_code=camera_code, data_time=data_time
        )
        if url:
            urls.append(url)
    return urls


def normalize_images(images: Optional[Sequence[str]]) -> List[str]:
    """清理图片 Base64 入参（去空、去首尾空白）。"""
    return [str(i).strip() for i in (images or []) if str(i or "").strip()]
