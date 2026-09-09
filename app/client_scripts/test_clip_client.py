"""
事件视频截取 - 客户端测试脚本

先启动服务端（在 code 目录下）:
  python -m app.main

确保 ZLMediaKit 已运行，且目标流在线（可先推流）:
  python ./app/client_scripts/test_stream_client.py start

再运行本脚本:
  python ./app/client_scripts/test_clip_client.py capture --id 1
  python ./app/client_scripts/test_clip_client.py capture --id 1 --video_url rtmp://10.1.3.21:1935/scene_001/person_count_detector26 --back_ms 10000 --forward_ms 10000
  python ./app/client_scripts/test_clip_client.py capture --id 1 --output_dir ./clip_output
  python ./app/client_scripts/test_clip_client.py demo --id 1
"""
import argparse
import base64
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

# client_scripts -> app -> code
CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))

from app.core.config import settings

DEFAULT_BASE_URL = settings.api_base_url
DEFAULT_OUTPUT_DIR = CODE_ROOT / "clip_output"
DEFAULT_VIDEO = settings.resolve_test_video_path(CODE_ROOT)
DEFAULT_STREAM_URL = settings.build_push_url()


class ClipServiceClient:
    """事件视频截取 HTTP 客户端"""

    def __init__(self, base_url: str = DEFAULT_BASE_URL, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(
        self,
        method: str,
        path: str,
        body: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        data = None
        headers = {"Accept": "application/json"}
        if body is not None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"

        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="ignore")
            try:
                err_json = json.loads(detail)
                detail = err_json.get("detail", detail)
            except json.JSONDecodeError:
                pass
            raise RuntimeError(f"HTTP {e.code} {e.reason}: {detail}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"无法连接服务 {url}: {e.reason}") from e

    def capture_clip(self, payload: Dict[str, Any], timeout: Optional[float] = None) -> Dict[str, Any]:
        return self._request("POST", "/clip/capture", body=payload, timeout=timeout)

    def start_stream(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "/streams/start", body=payload)

    def stop_stream(self, task_id: str) -> Dict[str, Any]:
        return self._request("POST", f"/streams/stop/{task_id}")

    def health(self) -> Dict[str, Any]:
        url = self.base_url.replace("/api/v1", "") + "/health"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            return json.loads(resp.read().decode("utf-8"))


def _print_json(data: Dict[str, Any]):
    print(json.dumps(data, ensure_ascii=False, indent=2))


def _build_capture_payload(args) -> Dict[str, Any]:
    video_url = args.video_url or DEFAULT_STREAM_URL
    return {
        "id": args.id,
        "video_url": video_url,
        "back_ms": args.back_ms,
        "forward_ms": args.forward_ms,
    }


def _estimate_timeout(back_ms: int, forward_ms: int, buffer_ms: int = 2000, extra: float = 30.0) -> float:
    return (back_ms + forward_ms + buffer_ms) / 1000.0 + extra


def _save_clip_assets(result: Dict[str, Any], output_dir: Path, prefix: Optional[str] = None) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    name_prefix = prefix or f"{result.get('app', 'clip')}_{result.get('stream', 'stream')}_{stamp}"

    video_path = output_dir / f"{name_prefix}.mp4"
    image_path = output_dir / f"{name_prefix}.jpg"
    meta_path = output_dir / f"{name_prefix}.json"

    video_b64 = result.get("video_base64") or ""
    image_b64 = result.get("image_base64") or ""
    if not video_b64 or not image_b64:
        raise RuntimeError("响应缺少 video_base64 或 image_base64")

    video_path.write_bytes(base64.b64decode(video_b64))
    image_path.write_bytes(base64.b64decode(image_b64))

    meta = {
        k: v
        for k, v in result.items()
        if k not in ("video_base64", "image_base64")
    }
    meta["saved_video"] = str(video_path)
    meta["saved_image"] = str(image_path)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "video": str(video_path),
        "image": str(image_path),
        "meta": str(meta_path),
    }


def _print_capture_summary(result: Dict[str, Any], saved: Optional[Dict[str, str]] = None):
    summary = {k: v for k, v in result.items() if k not in ("video_base64", "image_base64")}
    _print_json(summary)

    video_size = len(result.get("video_base64", "")) * 3 // 4
    image_size = len(result.get("image_base64", "")) * 3 // 4
    print(
        f"\n视频 Base64 约 {video_size / 1024:.1f} KB，"
        f"图片 Base64 约 {image_size / 1024:.1f} KB"
    )
    if saved:
        print(f"已保存视频: {saved['video']}")
        print(f"已保存图片: {saved['image']}")
        print(f"已保存元数据: {saved['meta']}")


def cmd_capture(client: ClipServiceClient, args):
    payload = _build_capture_payload(args)
    timeout = args.timeout or _estimate_timeout(args.back_ms, args.forward_ms)

    print("发送截取请求（同步等待录制完成，请耐心等待）:")
    _print_json(payload)
    print(f"\n请求超时设置: {timeout:.0f}s\n")

    started = time.time()
    result = client.capture_clip(payload, timeout=timeout)
    elapsed = time.time() - started
    print(f"截取完成，耗时 {elapsed:.1f}s")

    saved = None
    if not args.no_save:
        output_dir = Path(args.output_dir)
        saved = _save_clip_assets(result, output_dir)

    _print_capture_summary(result, saved)


def cmd_demo(client: ClipServiceClient, args):
    """启动推流 -> 等待流稳定 -> 截取视频与中间帧 -> 可选停止推流"""
    out_url = args.out_url or settings.build_push_url(
        args.output_format,
        scene_id=args.scene_id,
        skill_name=args.skill_name,
    )
    start_payload = {
        "in_url": args.in_url,
        "out_url": out_url,
        "out_fps": args.out_fps,
        "alarm_interval": args.alarm_interval,
        "scene_id": args.scene_id,
        "skill_name": args.skill_name,
        "output_format": args.output_format,
        "queue_size": args.queue_size,
        "use_hardware_encoding": not args.no_hw_encode,
        "bitrate": args.bitrate,
        "buffer_size": args.buffer_size,
        "reconnect_delay": args.reconnect_delay,
    }

    print("=== 演示：启动推流并截取事件视频 ===")
    print("1) 启动推流任务")
    _print_json(start_payload)
    start_result = client.start_stream(start_payload)
    task_id = start_result.get("task_id")
    _print_json(start_result)
    if not task_id:
        raise RuntimeError("推流任务启动失败，未返回 task_id")

    print(f"\n2) 等待 {args.warmup_sec}s，让流稳定并积累 GOP 缓存...")
    time.sleep(args.warmup_sec)

    capture_args = argparse.Namespace(
        video_url=args.video_url if args.video_url else out_url,
        back_ms=args.back_ms,
        forward_ms=args.forward_ms,
        timeout=args.timeout,
        output_dir=args.output_dir,
        no_save=args.no_save,
    )

    print("\n3) 调用事件视频截取")
    try:
        cmd_capture(client, capture_args)
    finally:
        if not args.keep_stream:
            print("\n4) 停止推流任务")
            stop_result = client.stop_stream(task_id)
            _print_json(stop_result)
        else:
            print(f"\n推流任务保持运行，task_id = {task_id}")


def _add_capture_args(parser: argparse.ArgumentParser):
    parser.add_argument("--base_url", default=DEFAULT_BASE_URL, help="API 基础地址")
    parser.add_argument("--id", type=int, required=True, help="告警记录 ID（回写 coal/alarm/update）")
    parser.add_argument(
        "--video_url",
        default=None,
        help=f"目标视频流地址（RTMP / RTSP / HTTP-FLV / HLS），默认 {DEFAULT_STREAM_URL}",
    )
    parser.add_argument("--back_ms", type=int, default=10000, help="回溯录制时长（毫秒）")
    parser.add_argument("--forward_ms", type=int, default=10000, help="后续录制时长（毫秒）")
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="HTTP 超时（秒）；默认按 back_ms + forward_ms 自动估算",
    )
    parser.add_argument(
        "--output_dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="保存视频/图片/元数据的目录",
    )
    parser.add_argument(
        "--no_save",
        action="store_true",
        help="不将 Base64 结果落盘，仅打印摘要",
    )


def _add_demo_stream_args(parser: argparse.ArgumentParser):
    parser.add_argument("--in_url", default=DEFAULT_VIDEO, help="演示推流输入视频源")
    parser.add_argument("--out_url", default=None, help="演示推流输出地址")
    parser.add_argument("--scene_id", default=settings.DEFAULT_SCENE_ID, help="场景 ID")
    parser.add_argument("--skill_name", default=settings.DEFAULT_SKILL_NAME, help="技能名称")
    parser.add_argument(
        "--output_format",
        default=settings.DEFAULT_OUTPUT_FORMAT,
        choices=["rtmp", "rtsp"],
    )
    parser.add_argument("--out_fps", type=int, default=settings.DEFAULT_OUT_FPS)
    parser.add_argument("--alarm_interval", type=int, default=settings.DEFAULT_ALARM_INTERVAL)
    parser.add_argument("--queue_size", type=int, default=settings.STREAM_QUEUE_SIZE)
    parser.add_argument("--no_hw_encode", action="store_true", help="禁用硬件编码")
    parser.add_argument("--bitrate", default=settings.STREAM_BITRATE)
    parser.add_argument("--buffer_size", default=settings.STREAM_BUFFER_SIZE)
    parser.add_argument("--reconnect_delay", type=float, default=settings.STREAM_RECONNECT_DELAY)
    parser.add_argument(
        "--warmup_sec",
        type=float,
        default=15.0,
        help="推流启动后等待秒数，便于积累回溯缓存",
    )
    parser.add_argument(
        "--keep_stream",
        action="store_true",
        help="演示结束后不停止推流任务",
    )


def main():
    parser = argparse.ArgumentParser(description="事件视频截取客户端测试")
    sub = parser.add_subparsers(dest="command", required=True)

    p_capture = sub.add_parser("capture", help="截取事件视频并保存中间帧图片")
    _add_capture_args(p_capture)

    p_demo = sub.add_parser("demo", help="启动推流后自动截取并保存")
    _add_capture_args(p_demo)
    _add_demo_stream_args(p_demo)

    args = parser.parse_args()
    client = ClipServiceClient(base_url=args.base_url)

    handlers = {
        "capture": cmd_capture,
        "demo": cmd_demo,
    }

    try:
        handlers[args.command](client, args)
    except RuntimeError as e:
        print(f"\n错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
