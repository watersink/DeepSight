"""
自动化测试入口 — 依次执行项目相关手工测试命令。

在 code 目录下运行:
  python app/auto_unit_test/run_tests.py
  python app/auto_unit_test/run_tests.py --only skill
  python app/auto_unit_test/run_tests.py --only server --only client

测试项:
  1. 技能测试       python app/plugins/skills/person_count_detector_skill.py
  2. 服务端测试     python -m app.main（短时启动并检查 /health）
  3. 客户端测试     python app/client_scripts/test_stream_client.py start ...
  4. 本地推流测试   python app/client_scripts/test_stream_process.py（短时运行后停止）
"""
from __future__ import annotations

import argparse
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# app/auto_unit_test -> code
CODE_ROOT = Path(__file__).resolve().parent.parent.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

SKILL_SCRIPT = CODE_ROOT / "app" / "plugins" / "skills" / "person_count_detector_skill.py"
STREAM_PROCESS_SCRIPT = CODE_ROOT / "app" / "client_scripts" / "test_stream_process.py"
STREAM_CLIENT_SCRIPT = CODE_ROOT / "app" / "client_scripts" / "test_stream_client.py"
TEST_VIDEO = CODE_ROOT / "test_images_videos" / "person_gouzi.mp4"
SKILL_IMAGE = CODE_ROOT / "test_images_videos" / "9-74-0004-000880.jpg"

DEFAULT_SCENE_ID = "scene_001"
DEFAULT_SKILL_NAME = "person_count_detector"
STREAM_RUN_SECONDS = 8
SERVER_START_TIMEOUT = 30


@dataclass
class CaseResult:
    name: str
    passed: bool
    message: str = ""
    skipped: bool = False


@dataclass
class RunReport:
    results: List[CaseResult] = field(default_factory=list)

    def add(self, result: CaseResult) -> None:
        self.results.append(result)

    def print_summary(self) -> int:
        print("\n" + "=" * 60)
        print("测试汇总")
        print("=" * 60)
        for r in self.results:
            if r.skipped:
                tag = "SKIP"
            elif r.passed:
                tag = "PASS"
            else:
                tag = "FAIL"
            print(f"[{tag}] {r.name}")
            if r.message:
                for line in r.message.strip().splitlines():
                    print(f"       {line}")
        passed = sum(1 for r in self.results if r.passed and not r.skipped)
        failed = sum(1 for r in self.results if not r.passed and not r.skipped)
        skipped = sum(1 for r in self.results if r.skipped)
        print("-" * 60)
        print(f"通过: {passed}  失败: {failed}  跳过: {skipped}  合计: {len(self.results)}")
        return 1 if failed else 0


def _pick_free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]


def _wait_http(url: str, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, TimeoutError):
            time.sleep(0.5)
    return False


def _run_cmd(
    args: List[str],
    *,
    timeout: Optional[float] = None,
    cwd: Path = CODE_ROOT,
) -> subprocess.CompletedProcess:
    print(f"\n>>> {' '.join(args)}")
    return subprocess.run(
        args,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _stop_proc(proc: subprocess.Popen, grace: float = 5.0) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def _should_skip_external_error(text: str) -> bool:
    markers = (
        "无法连接",
        "Connection refused",
        "UNAVAILABLE",
        "Failed to connect",
        "Triton",
    )
    return any(m in text for m in markers)


def test_skill(report: RunReport) -> None:
    name = "技能测试 (person_count_detector_skill.py)"
    if not SKILL_SCRIPT.is_file():
        report.add(CaseResult(name, False, f"脚本不存在: {SKILL_SCRIPT}", skipped=True))
        return
    if not SKILL_IMAGE.is_file():
        report.add(CaseResult(name, False, f"测试图片不存在: {SKILL_IMAGE}", skipped=True))
        return

    try:
        result = _run_cmd([sys.executable, str(SKILL_SCRIPT)], timeout=120)
    except subprocess.TimeoutExpired:
        report.add(CaseResult(name, False, "执行超时（120s）"))
        return

    output = f"{result.stdout}\n{result.stderr}"
    if result.returncode != 0:
        if _should_skip_external_error(output):
            report.add(CaseResult(name, False, f"Triton 不可用，已跳过\n{output}", skipped=True))
        else:
            report.add(CaseResult(name, False, output))
        return

    report.add(CaseResult(name, True, result.stdout.strip() or "技能脚本执行成功"))


def _start_server(port: int) -> subprocess.Popen:
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=str(CODE_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    health_url = f"http://127.0.0.1:{port}/health"
    if not _wait_http(health_url, SERVER_START_TIMEOUT):
        _stop_proc(proc)
        out, err = proc.communicate(timeout=3)
        raise RuntimeError(f"服务启动失败\nstdout:\n{out}\nstderr:\n{err}")
    return proc


def test_server(report: RunReport) -> Optional[int]:
    """启动服务端并检查 /health，返回使用的端口（服务保持运行供客户端测试）。"""
    name = "服务端测试 (app.main /health)"
    port = _pick_free_port()
    try:
        proc = _start_server(port)
    except RuntimeError as e:
        report.add(CaseResult(name, False, str(e)))
        return None

    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as resp:
            data = resp.read().decode("utf-8")
        if '"status"' in data and "ok" in data:
            report._server_proc = proc  # noqa: SLF001
            report.add(CaseResult(name, True, f"服务已启动 port={port}\n{data.strip()}"))
            return port
        report.add(CaseResult(name, False, f"健康检查响应异常: {data}"))
    except Exception as e:
        report.add(CaseResult(name, False, str(e)))
    finally:
        if not any(r.name == name and r.passed for r in report.results):
            _stop_proc(proc)
    return None


def test_client(report: RunReport, port: Optional[int] = None) -> None:
    name = "客户端测试 (test_stream_client.py start)"
    if not STREAM_CLIENT_SCRIPT.is_file():
        report.add(CaseResult(name, False, f"脚本不存在: {STREAM_CLIENT_SCRIPT}", skipped=True))
        return
    if not TEST_VIDEO.is_file():
        report.add(CaseResult(name, False, f"测试视频不存在: {TEST_VIDEO}", skipped=True))
        return

    server_proc = getattr(report, "_server_proc", None)
    own_server = False
    if port is None:
        port = _pick_free_port()
        try:
            server_proc = _start_server(port)
            own_server = True
        except RuntimeError as e:
            report.add(CaseResult(name, False, f"无法启动 API 服务: {e}"))
            return

    base_url = f"http://127.0.0.1:{port}/api/v1"
    rel_video = TEST_VIDEO.relative_to(CODE_ROOT)

    try:
        start = _run_cmd(
            [
                sys.executable,
                str(STREAM_CLIENT_SCRIPT),
                "start",
                "--base_url",
                base_url,
                "--in_url",
                str(rel_video),
                "--scene_id",
                DEFAULT_SCENE_ID,
                "--skill_name",
                DEFAULT_SKILL_NAME,
                "--no_hw_encode",
                "--out_fps",
                "5",
            ],
            timeout=90,
        )
    except subprocess.TimeoutExpired:
        report.add(CaseResult(name, False, "客户端 start 超时（90s）"))
        return
    finally:
        if own_server and server_proc:
            _stop_proc(server_proc)

    output = f"{start.stdout}\n{start.stderr}"
    if start.returncode != 0:
        if _should_skip_external_error(output):
            report.add(CaseResult(name, False, f"外部依赖不可用，已跳过\n{output}", skipped=True))
        else:
            report.add(CaseResult(name, False, output))
        return

    match = re.search(r'"task_id"\s*:\s*"([^"]+)"', start.stdout)
    task_id = match.group(1) if match else None
    if not task_id:
        for line in start.stdout.splitlines():
            if "task_id" in line and "=" in line:
                task_id = line.split("=")[-1].strip()
                break

    if not task_id:
        report.add(CaseResult(name, False, f"未解析到 task_id:\n{start.stdout}"))
        return

    stop = _run_cmd(
        [
            sys.executable,
            str(STREAM_CLIENT_SCRIPT),
            "stop",
            "--base_url",
            base_url,
            "--task_id",
            task_id,
        ],
        timeout=30,
    )
    if stop.returncode != 0:
        report.add(CaseResult(name, False, f"启动成功但停止失败 task_id={task_id}\n{stop.stderr}"))
        return

    report.add(CaseResult(name, True, f"task_id={task_id}\n{start.stdout.strip()}"))


def test_stream_process(report: RunReport) -> None:
    name = "本地推流测试 (test_stream_process.py)"
    if not STREAM_PROCESS_SCRIPT.is_file():
        report.add(CaseResult(name, False, f"脚本不存在: {STREAM_PROCESS_SCRIPT}", skipped=True))
        return
    if not TEST_VIDEO.is_file():
        report.add(CaseResult(name, False, f"测试视频不存在: {TEST_VIDEO}", skipped=True))
        return

    print(f"\n>>> 短时运行推流脚本（{STREAM_RUN_SECONDS}s 后自动停止）")
    proc = subprocess.Popen(
        [
            sys.executable,
            str(STREAM_PROCESS_SCRIPT),
            "--in_url",
            str(TEST_VIDEO),
            "--scene_id",
            DEFAULT_SCENE_ID,
            "--skill_name",
            DEFAULT_SKILL_NAME,
            "--no_hw_encode",
            "--out_fps",
            "5",
        ],
        cwd=str(CODE_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        time.sleep(STREAM_RUN_SECONDS)
        if proc.poll() is not None:
            out, err = proc.communicate(timeout=5)
            combined = f"{out}\n{err}"
            if _should_skip_external_error(combined):
                report.add(CaseResult(name, False, f"外部依赖不可用，已跳过\n{combined}", skipped=True))
            else:
                report.add(CaseResult(name, False, f"进程过早退出 code={proc.returncode}\n{combined}"))
            return
        report.add(CaseResult(name, True, f"推流进程已运行 {STREAM_RUN_SECONDS}s 后正常停止"))
    finally:
        _stop_proc(proc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Video Detection Stream 自动化测试")
    parser.add_argument(
        "--only",
        action="append",
        choices=["skill", "server", "client", "stream"],
        help="仅运行指定测试（可多次指定）",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    order = args.only or ["skill", "server", "client", "stream"]
    report = RunReport()

    print(f"工作目录: {CODE_ROOT}")
    print(f"计划执行: {', '.join(order)}")

    server_port: Optional[int] = None
    for item in order:
        if item == "skill":
            test_skill(report)
        elif item == "server":
            server_port = test_server(report)
        elif item == "client":
            test_client(report, server_port)
            proc = getattr(report, "_server_proc", None)
            if proc:
                _stop_proc(proc)
                report._server_proc = None  # noqa: SLF001
        elif item == "stream":
            test_stream_process(report)

    proc = getattr(report, "_server_proc", None)
    if proc:
        _stop_proc(proc)

    return report.print_summary()


if __name__ == "__main__":
    sys.exit(main())
