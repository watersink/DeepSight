"""
FFmpeg 实时帧推流器 - 参考 smart_engine FFmpegFrameStreamer

支持 RTMP (FLV) 与 RTSP 两种输出，统一进程管理与自动重启。
"""
import atexit
import logging
import subprocess
import threading
import time
import weakref
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Set

import cv2
import numpy as np

from app.services.video.encoder import get_encoder_options

logger = logging.getLogger(__name__)

_active_streamers: Set[weakref.ref] = set()
_cleanup_lock = threading.Lock()


def _register_streamer(streamer):
    with _cleanup_lock:
        _active_streamers.add(weakref.ref(streamer))


def _unregister_streamer(streamer):
    with _cleanup_lock:
        _active_streamers.discard(weakref.ref(streamer))


def cleanup_all_streamers():
    with _cleanup_lock:
        for ref in list(_active_streamers):
            s = ref()
            if s is not None:
                try:
                    s.stop()
                except Exception as e:
                    logger.warning("清理推流器时出错: %s", e)
        _active_streamers.clear()


atexit.register(cleanup_all_streamers)


def _ensure_even_size(width: int, height: int) -> tuple:
    """H.264/yuv420p 要求宽高为偶数"""
    return max(2, width - width % 2), max(2, height - height % 2)


class FFmpegStreamerBase(ABC):
    """FFmpeg 推流器基类"""

    STDERR_MAX_LINES = 50

    def __init__(
        self,
        output_url: str,
        fps: float,
        width: int,
        height: int,
        use_hardware_encoding: bool = True,
        bitrate: str = "2M",
        buffer_size: str = "2M",
    ):
        self.output_url = output_url
        self.fps = max(fps, 1.0)
        self.width, self.height = _ensure_even_size(width, height)
        self.use_hardware_encoding = use_hardware_encoding
        self.bitrate = bitrate
        self.buffer_size = buffer_size

        self.process: Optional[subprocess.Popen] = None
        self.is_running = False
        self.lock = threading.Lock()

        self.restart_count = 0
        self.max_restart_attempts = 5
        self.last_restart_time = 0.0
        self.restart_interval = 10.0

        self.stats: Dict[str, Any] = {
            "frames_sent": 0,
            "errors": 0,
            "restarts": 0,
            "last_error": None,
        }

        self._stderr_lines: list = []
        self._stderr_lock = threading.Lock()
        self._stderr_reader_thread: Optional[threading.Thread] = None

    def get_encoder_options(self) -> list:
        return get_encoder_options(
            self.fps,
            self.use_hardware_encoding,
            self.bitrate,
            self.buffer_size,
        )

    @abstractmethod
    def _build_ffmpeg_command(self) -> list:
        pass

    def start(self) -> bool:
        with self.lock:
            if self.is_running:
                return True
            return self._start_process()

    def _start_stderr_reader(self):
        if self.process and self.process.stderr:
            self._stderr_reader_thread = threading.Thread(
                target=self._stderr_reader_worker,
                daemon=True,
                name=f"FFmpegStderr-{id(self)}",
            )
            self._stderr_reader_thread.start()

    def _stderr_reader_worker(self):
        """持续读取 stderr，避免管道缓冲区满导致 FFmpeg 阻塞"""
        try:
            proc = self.process
            if not proc or not proc.stderr:
                return
            for line_bytes in iter(proc.stderr.readline, b""):
                if not self.is_running and proc.poll() is not None:
                    break
                line = line_bytes.decode("utf-8", errors="ignore").rstrip()
                if not line:
                    continue
                with self._stderr_lock:
                    self._stderr_lines.append({"time": time.time(), "message": line})
                    if len(self._stderr_lines) > self.STDERR_MAX_LINES:
                        self._stderr_lines = self._stderr_lines[-self.STDERR_MAX_LINES:]
                lower = line.lower()
                if any(kw in lower for kw in ("error", "failed", "refused", "denied", "invalid")):
                    logger.warning("FFmpeg: %s", line[:300])
                    self.stats["last_error"] = line[:500]
        except Exception:
            pass

    def _get_stderr_snapshot(self) -> str:
        with self._stderr_lock:
            if not self._stderr_lines:
                return ""
            return "\n".join(item["message"] for item in self._stderr_lines[-20:])

    def _start_process(self) -> bool:
        try:
            cmd = self._build_ffmpeg_command()
            logger.info("启动 FFmpeg 推流: %s", self.output_url)
            logger.info("FFmpeg 命令: %s", " ".join(cmd))

            flags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                bufsize=0,
                creationflags=flags,
            )
            self._start_stderr_reader()
            time.sleep(0.5)

            if self.process.poll() is not None:
                stderr = self._get_stderr_snapshot() or self._read_stderr()
                logger.error("FFmpeg 启动失败: %s", stderr[:500])
                self.stats["last_error"] = stderr[:500]
                return False

            self.is_running = True
            self.restart_count = 0
            _register_streamer(self)
            return True
        except FileNotFoundError:
            logger.error("FFmpeg 未安装或不在 PATH 中")
            return False
        except Exception as e:
            logger.error("启动推流失败: %s", e)
            self.stats["last_error"] = str(e)
            return False

    def _read_stderr(self) -> str:
        if not self.process or not self.process.stderr:
            return ""
        try:
            return self.process.stderr.read().decode("utf-8", errors="ignore")
        except Exception:
            return ""

    def stop(self):
        with self.lock:
            self.is_running = False
            _unregister_streamer(self)
            if not self.process:
                return
            try:
                if self.process.stdin:
                    self.process.stdin.close()
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.process.kill()
            except Exception as e:
                logger.warning("停止 FFmpeg 时出错: %s", e)
            finally:
                self.process = None

        if (
            self._stderr_reader_thread
            and self._stderr_reader_thread.is_alive()
            and self._stderr_reader_thread is not threading.current_thread()
        ):
            self._stderr_reader_thread.join(timeout=2)
        self._stderr_reader_thread = None

    def reset_restart_count(self):
        self.restart_count = 0

    def _should_restart(self) -> bool:
        if self.restart_count >= self.max_restart_attempts:
            return False
        return time.time() - self.last_restart_time >= self.restart_interval

    def _restart(self) -> bool:
        self.stop()
        self.restart_count += 1
        self.last_restart_time = time.time()
        self.stats["restarts"] += 1
        logger.info("重启推流器(第 %d 次): %s", self.restart_count, self.output_url)
        return self._start_process()

    def push_frame(self, frame: np.ndarray) -> bool:
        try:
            if not self.is_running or not self.process:
                if self._should_restart() and self._restart():
                    return self.push_frame(frame)
                return False

            if self.process.poll() is not None:
                if self._should_restart() and self._restart():
                    return self.push_frame(frame)
                self.is_running = False
                return False

            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(frame, (self.width, self.height))

            self.process.stdin.write(frame.tobytes())
            self.process.stdin.flush()
            self.stats["frames_sent"] += 1
            return True

        except BrokenPipeError:
            self.stats["errors"] += 1
            if self._should_restart() and self._restart():
                return self.push_frame(frame)
            self.is_running = False
            return False
        except OSError as e:
            self.stats["errors"] += 1
            self.stats["last_error"] = str(e)
            if self._should_restart() and self._restart():
                return self.push_frame(frame)
            self.is_running = False
            return False
        except Exception as e:
            logger.error("推送帧失败: %s", e)
            self.stats["errors"] += 1
            return False


class FFmpegRTMPStreamer(FFmpegStreamerBase):
    """RTMP (FLV) 推流"""

    def _build_ffmpeg_command(self) -> list:
        cmd = [
            "ffmpeg", "-y",
            "-loglevel", "warning",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{self.width}x{self.height}",
            "-r", str(self.fps),
            "-thread_queue_size", "512",
            "-i", "-",
        ]
        cmd.extend(self.get_encoder_options())
        cmd.extend([
            "-an",  # 无音频轨，避免部分播放器等待音频
            "-f", "flv",
            "-flvflags", "no_duration_filesize",  # 直播流必须，否则播放器可能无法播放
            self.output_url,
        ])
        return cmd


class FFmpegRTSPStreamer(FFmpegStreamerBase):
    """RTSP 推流"""

    def _build_ffmpeg_command(self) -> list:
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{self.width}x{self.height}",
            "-r", str(self.fps),
            "-thread_queue_size", "512",
            "-i", "-",
        ]
        cmd.extend(self.get_encoder_options())
        cmd.extend([
            "-f", "rtsp",
            "-rtsp_transport", "tcp",
            "-timeout", "5000000",
            self.output_url,
        ])
        return cmd


def create_streamer(
    output_url: str,
    fps: float,
    width: int,
    height: int,
    output_format: str = "rtmp",
    **kwargs,
) -> FFmpegStreamerBase:
    """工厂方法：按输出格式创建推流器"""
    fmt = output_format.lower()
    if fmt == "rtsp":
        return FFmpegRTSPStreamer(output_url, fps, width, height, **kwargs)
    return FFmpegRTMPStreamer(output_url, fps, width, height, **kwargs)
