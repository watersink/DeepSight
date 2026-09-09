"""
视频流推流服务 - 客户端测试脚本

先启动服务端（在 code 目录下）:
  python -m app.main

再运行本脚本:
  python ./app/client_scripts/test_stream_client.py start
  python ./app/client_scripts/test_stream_client.py start --in_url rtmp://10.1.3.21:1935/live/stream01 --skill_name person_count_detector26 --scene_id scene_001 --enter_count 0 --gate_direction IN --count_line ./app/client_scripts/config/person_count_detector26_count_line.json --bypass_line ./app/client_scripts/config/person_count_detector26_bypass_line.json
  python ./app/client_scripts/test_stream_client.py start --in_url rtmp://10.1.3.21:1935/live/stream01 --skill_name person_count_detector26 --scene_id scene_002 --enter_count 0 --gate_direction IN --count_line ./app/client_scripts/config/person_count_detector26_count_line_monkeycar.json
  python ./app/client_scripts/test_stream_client.py start --in_url rtmp://10.1.3.21:1935/live/stream01 --skill_name person_count_detector26 --scene_id scene_002 --enter_count 0 --gate_direction IN --count_line ./app/client_scripts/config/person_count_detector26_count_line_guanlongin1.json
  python ./app/client_scripts/test_stream_client.py start --in_url rtmp://10.1.3.21:1935/live/stream01 --skill_name person_presence_detector26 --scene_id scene_003
  python ./app/client_scripts/test_stream_client.py start --in_url ./test_images_videos/person_gouzi.mp4 --scene_id scene_001
  python ./app/client_scripts/test_stream_client.py start --skill_name person_count_detector26 --in_url ./test_images_videos/person_gouzi.mp4
  python ./app/client_scripts/test_stream_client.py status --task_id <task_id>
  python ./app/client_scripts/test_stream_client.py stop --task_id <task_id>
  python ./app/client_scripts/test_stream_client.py list
  python ./app/client_scripts/test_stream_client.py skills
  python ./app/client_scripts/test_stream_client.py health
  python ./app/client_scripts/test_stream_client.py demo
"""
import argparse
import json
import re
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
DEFAULT_VIDEO = settings.resolve_test_video_path(CODE_ROOT)


class StreamServiceClient:
    """推流服务 HTTP 客户端"""

    def __init__(self, base_url: str = DEFAULT_BASE_URL, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(
        self,
        method: str,
        path: str,
        body: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        data = None
        headers = {"Accept": "application/json"}
        if body is not None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"

        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
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

    def health(self) -> Dict[str, Any]:
        url = self.base_url.replace("/api/v1", "") + "/health"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def start_stream(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "/streams/start", body=payload)

    def stop_stream(self, task_id: str) -> Dict[str, Any]:
        return self._request("POST", f"/streams/stop/{task_id}")

    def get_stream(self, task_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/streams/{task_id}")

    def list_streams(self) -> Dict[str, Any]:
        return self._request("GET", "/streams")

    def list_skills(self) -> Dict[str, Any]:
        return self._request("GET", "/skills")


def _print_json(data: Dict[str, Any]):
    print(json.dumps(data, ensure_ascii=False, indent=2))


def _parse_json_arg(raw: str, arg_name: str) -> Any:
    """
    解析命令行 JSON 参数，兼容 Windows PowerShell 吃掉双引号的情况。

    也支持传入 .json 文件路径。
    """
    text = (raw or "").strip()
    if not text:
        raise ValueError(f"{arg_name} 为空")

    path = Path(text)
    if path.is_file():
        text = path.read_text(encoding="utf-8").strip()
    elif text.lower().endswith(".json") or "/" in text or "\\" in text:
        raise ValueError(
            f"{arg_name} 路径不存在: {path.resolve() if not path.is_absolute() else path}\n"
            f"请确认文件名（当前目录下可用的类似配置见 app/client_scripts/config/）"
        )

    candidates = [text]
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"":
        candidates.append(text[1:-1].strip())

    # PowerShell 常见结果: {line:[...],enter_point:{...}} —— 给裸 key 补双引号
    def _quote_bare_keys(s: str) -> str:
        return re.sub(r'([{\[,]\s*)([A-Za-z_][\w]*)\s*:', r'\1"\2":', s)

    for candidate in list(candidates):
        candidates.append(candidate.replace("'", '"'))
        candidates.append(_quote_bare_keys(candidate))
        candidates.append(_quote_bare_keys(candidate.replace("'", '"')))

    last_error: Optional[Exception] = None
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as e:
            last_error = e

    raise ValueError(
        f"无法解析 {arg_name} 为 JSON。PowerShell 请用:\n"
        f'  --{arg_name} "{{\\"line\\":[{{\\"x\\":432,\\"y\\":328}},{{\\"x\\":723,\\"y\\":229}}],'
        f'\\"enter_point\\":{{\\"x\\":513,\\"y\\":250}}}}"\n'
        f"或先写入文件再传路径。原始值={raw!r} 错误={last_error}"
    ) from last_error


def _build_start_payload(args) -> Dict[str, Any]:
    output_format = args.output_format
    out_url = args.out_url
    if not out_url:
        out_url = settings.build_push_url(
            output_format,
            scene_id=args.scene_id,
            skill_name=args.skill_name,
        )

    payload = {
        "in_url": args.in_url,
        "out_url": out_url,
        "out_fps": args.out_fps,
        "alarm_interval": args.alarm_interval,
        "scene_id": args.scene_id,
        "skill_name": args.skill_name,
        "output_format": output_format,
        "queue_size": args.queue_size,
        "use_hardware_encoding": not args.no_hw_encode,
        "bitrate": args.bitrate,
        "buffer_size": args.buffer_size,
        "reconnect_delay": args.reconnect_delay,
    }
    if args.fence_config:
        payload["fence_config"] = _parse_json_arg(args.fence_config, "fence_config")
    if args.skill_config:
        payload["skill_config"] = _parse_json_arg(args.skill_config, "skill_config")
    if getattr(args, "enter_count", None) is not None:
        payload["enter_count"] = int(args.enter_count)
    if getattr(args, "gate_direction", None):
        direction = str(args.gate_direction).strip().upper()
        if direction not in {"IN", "OUT"}:
            raise ValueError(f"--gate_direction 仅支持 IN/OUT，收到: {args.gate_direction}")
        payload["gate_direction"] = direction
    if getattr(args, "count_line", None):
        payload["count_line"] = _parse_json_arg(args.count_line, "count_line")
    if getattr(args, "bypass_line", None):
        payload["bypass_line"] = _parse_json_arg(args.bypass_line, "bypass_line")
    if getattr(args, "mine_code", None):
        payload["mine_code"] = str(args.mine_code).strip()
    if getattr(args, "camera_code", None):
        payload["camera_code"] = str(args.camera_code).strip()
    return payload


def cmd_health(client: StreamServiceClient, _args):
    result = client.health()
    _print_json(result)
    print("\n服务可用。" if result.get("status") == "ok" else "\n服务异常。")


def cmd_start(client: StreamServiceClient, args):
    payload = _build_start_payload(args)
    print("发送启动请求:")
    _print_json(payload)
    result = client.start_stream(payload)
    print("\n启动成功:")
    _print_json(result)
    print(f"\ntask_id = {result.get('task_id')}")
    print("查询状态: python ./app/client_scripts/test_stream_client.py status --task_id", result.get("task_id"))
    print("停止任务: python ./app/client_scripts/test_stream_client.py stop --task_id", result.get("task_id"))


def cmd_stop(client: StreamServiceClient, args):
    if not args.task_id:
        print("请提供 --task_id")
        sys.exit(1)
    result = client.stop_stream(args.task_id)
    print("任务已停止:")
    _print_json(result)


def cmd_status(client: StreamServiceClient, args):
    if not args.task_id:
        print("请提供 --task_id")
        sys.exit(1)
    result = client.get_stream(args.task_id)
    _print_json(result)


def cmd_list(client: StreamServiceClient, _args):
    result = client.list_streams()
    _print_json(result)


def cmd_skills(client: StreamServiceClient, _args):
    result = client.list_skills()
    _print_json(result)


def cmd_demo(client: StreamServiceClient, args):
    """启动任务并轮询状态，Ctrl+C 时自动停止"""
    payload = _build_start_payload(args)
    print("=== 演示：启动视频流识别任务 ===")
    _print_json(payload)

    result = client.start_stream(payload)
    task_id = result.get("task_id")
    print("\n任务已启动:")
    _print_json(result)

    if not task_id:
        sys.exit(1)

    print(f"\n每 {args.poll_interval}s 查询一次状态，按 Ctrl+C 停止任务...\n")
    try:
        while True:
            status = client.get_stream(task_id)
            print(
                f"[{time.strftime('%H:%M:%S')}] "
                f"task_id={task_id} status={status.get('status')} pid={status.get('pid')}"
            )
            if status.get("status") in ("stopped", "failed"):
                print("\n任务已结束:")
                _print_json(status)
                break
            time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        print("\n收到中断，正在停止任务...")
        stop_result = client.stop_stream(task_id)
        _print_json(stop_result)


def _add_common_start_args(parser: argparse.ArgumentParser):
    parser.add_argument("--base_url", default=DEFAULT_BASE_URL, help="API 基础地址")
    parser.add_argument("--in_url", default=DEFAULT_VIDEO, help="输入视频源")
    parser.add_argument(
        "--out_url",
        default=None,
        help=(
            "输出推流地址（默认: "
            f"{settings.build_push_url()}"
            "，即 {scene_id}/{skill_name}）"
        ),
    )
    parser.add_argument("--out_fps", type=int, default=settings.DEFAULT_OUT_FPS, help="输出帧率")
    parser.add_argument(
        "--alarm_interval",
        type=int,
        default=settings.DEFAULT_ALARM_INTERVAL,
        help="告警间隔（秒）",
    )
    parser.add_argument("--scene_id", default=settings.DEFAULT_SCENE_ID, help="场景 ID")
    parser.add_argument(
        "--skill_name",
        default=settings.DEFAULT_SKILL_NAME,
        help="技能名称，如 person_count_detector / person_count_detector26 / person_presence_detector26",
    )
    parser.add_argument(
        "--output_format",
        default=settings.DEFAULT_OUTPUT_FORMAT,
        choices=["rtmp", "rtsp"],
    )
    parser.add_argument("--queue_size", type=int, default=settings.STREAM_QUEUE_SIZE)
    parser.add_argument("--no_hw_encode", action="store_true", help="禁用硬件编码")
    parser.add_argument("--bitrate", default=settings.STREAM_BITRATE)
    parser.add_argument("--buffer_size", default=settings.STREAM_BUFFER_SIZE)
    parser.add_argument("--reconnect_delay", type=float, default=settings.STREAM_RECONNECT_DELAY)
    parser.add_argument("--fence_config", default=None, help="电子围栏 JSON 字符串")
    parser.add_argument("--skill_config", default=None, help="技能覆盖配置 JSON 字符串")
    parser.add_argument(
        "--enter_count",
        type=int,
        default=None,
        help="过线进入人数初始值（可选，部分技能如 person_count_detector26 使用，默认不传）",
    )
    parser.add_argument(
        "--gate_direction",
        default=None,
        choices=["IN", "OUT", "in", "out"],
        help="闸机方向：IN=入闸机口，OUT=出闸机口（可选，影响画面参考点标注）",
    )
    parser.add_argument(
        "--count_line",
        default=None,
        help="过线计数线 JSON 或 .json 文件路径（可选，像素坐标，lines 与 enter_point 一一对应）",
    )
    parser.add_argument(
        "--bypass_line",
        default=None,
        help="非计数绕行线 JSON 或 .json 文件路径（可选，像素坐标，lines 与 bypass_point 一一对应）",
    )
    parser.add_argument(
        "--mine_code",
        default=None,
        help="煤矿编码（12位数字，可选；用于识别结果上传）",
    )
    parser.add_argument(
        "--camera_code",
        default=None,
        help="摄像仪编码（20位，可选；用于识别结果上传）",
    )


def main():
    parser = argparse.ArgumentParser(description="AI 视频识别任务客户端测试")
    sub = parser.add_subparsers(dest="command", required=True)

    p_health = sub.add_parser("health", help="健康检查")
    p_health.add_argument("--base_url", default=DEFAULT_BASE_URL)

    p_start = sub.add_parser("start", help="启动 AI 视频识别任务")
    _add_common_start_args(p_start)

    p_stop = sub.add_parser("stop", help="停止 AI 视频识别任务")
    p_stop.add_argument("--base_url", default=DEFAULT_BASE_URL)
    p_stop.add_argument("--task_id", required=True)

    p_status = sub.add_parser("status", help="查询单个 AI 视频识别任务")
    p_status.add_argument("--base_url", default=DEFAULT_BASE_URL)
    p_status.add_argument("--task_id", required=True)

    p_list = sub.add_parser("list", help="查询所有 AI 视频识别任务")
    p_list.add_argument("--base_url", default=DEFAULT_BASE_URL)

    p_skills = sub.add_parser("skills", help="列出可用技能")
    p_skills.add_argument("--base_url", default=DEFAULT_BASE_URL)

    p_demo = sub.add_parser("demo", help="启动 AI 视频识别任务并轮询状态，Ctrl+C 停止")
    _add_common_start_args(p_demo)
    p_demo.add_argument("--poll_interval", type=float, default=5.0, help="状态轮询间隔（秒）")

    args = parser.parse_args()
    client = StreamServiceClient(base_url=args.base_url)

    handlers = {
        "health": cmd_health,
        "start": cmd_start,
        "stop": cmd_stop,
        "status": cmd_status,
        "list": cmd_list,
        "skills": cmd_skills,
        "demo": cmd_demo,
    }

    try:
        handlers[args.command](client, args)
    except (RuntimeError, ValueError) as e:
        print(f"\n错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
