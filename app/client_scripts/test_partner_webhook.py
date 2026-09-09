"""
第三方平台 Webhook 接收测试脚本

模拟对方系统：监听本机 HTTP，接收告警/事件通知后：
  1) 校验 X-Signature
  2) 拉取完整告警 JSON
  3) 下载告警图片
  4) 下载告警视频（未就绪时按策略重试；事件默认无视频）

---------------------------------------------------------------------------
第三方对接接口说明（本项目提供给对方调用 / 对方需提供）
鉴权：除「对方 Webhook」外，Open API 均需 Header: X-API-Key: <api_key>
基址：{OPEN_BASE}  例如 http://127.0.0.1:8000
---------------------------------------------------------------------------

【A. 对方必须实现的接口 — 接收实时通知】
  POST  {对方登记的 webhook_url}
    作用: 本项目在告警/事件落库后，按订阅过滤推送轻量通知（门铃）。
    请求头:
      Content-Type: application/json; charset=utf-8
      X-Timestamp: Unix 秒级时间戳
      X-Signature: hex(HMAC-SHA256(webhook_secret, "{X-Timestamp}.{raw_body}"))
      X-Partner-Code: 接入方编码
    请求体示例（轻量，不含大图/视频）:
      {
        "schema_version": "1",
        "event_type": "alert.created",   # 或 alert.test
        "alert_id": "<alert_uid>",
        "id": 123,
        "category": "alert" | "event",
        "recognition_types": ["04"],
        "scene_id": "scene_001",
        "skill_name": "...",
        "status": "new",
        "occurred_at": "2026-09-08T12:00:00+08:00",
        "has_image": true,
        "has_video": false,
        "href": "/open/v1/alerts/<alert_uid>"
      }
    响应: 尽快返回 2xx；业务处理可异步。本脚本监听路径:
      http://127.0.0.1:19090/webhook
      http://127.0.0.1:19090/api/alerts/webhook

【B. 本项目 Open API — 对方主动拉取】

  GET  {OPEN_BASE}/open/v1/health
    作用: 连通性探测，无需 API Key。

  GET  {OPEN_BASE}/open/v1/me
    作用: 查看当前接入方名称、订阅类别/类型/场景、是否配置了 Webhook。

  GET  {OPEN_BASE}/open/v1/alerts
    作用: 分页查询告警/事件列表（自动按该接入方订阅范围过滤）。
    常用 Query:
      page, page_size
      since 可用 time_from / time_to（ISO 时间）
      scene_id, status, recognition_type, category(alert|event)
    用途: 对方宕机恢复后补数、对账；也可不依赖 Webhook 纯轮询。

  GET  {OPEN_BASE}/open/v1/alerts/{alert_id}
    作用: 按 alert_uid（或数字 id）拉取完整告警 JSON（详情）。
    用途: 收到 Webhook 后，用通知里的 alert_id 拉全量字段。
    本脚本对应步骤: 保存 alert.json

  GET  {OPEN_BASE}/open/v1/alerts/{alert_id}/media/image
    作用: 获取证据图片；成功时 302 到对象存储 URL。
    用途: 展示/存证。无图返回 404。
    本脚本对应步骤: 保存 alert_image.*

  GET  {OPEN_BASE}/open/v1/alerts/{alert_id}/media/video
    作用: 获取证据短视频；成功时 302 到对象存储 URL。
    用途: 仅违规报警且任务开启「报警视频」时才有；事件通常无视频→404。
    未就绪也可 404，可稍后重试或等 has_video / media.ready。
    本脚本对应步骤: 保存 alert_video.*（event 会跳过）

  PATCH  {OPEN_BASE}/open/v1/alerts/{alert_id}/ack
    作用: 对方处理完成后回写状态。
    请求体: {"status": "acked" | "closed" | "new"}
    用途: 对账、避免重复处理（本测试脚本默认不调用）。

【C. 推荐调用顺序】
  1. 对方提供 webhook_url，在管理台「系统管理 → 第三方接入配置」登记
  2. 本项目 POST 对方 Webhook（轻量通知）
  3. 对方验签通过后 GET /open/v1/alerts/{alert_id} 取 JSON
  4. 若 has_image → GET .../media/image
  5. 若 has_video（且多为 category=alert）→ GET .../media/video
  6. （可选）PATCH .../ack 回写状态

【D. 本脚本启动示例】
  python ./app/client_scripts/test_partner_webhook.py ^
    --port 19090 ^
    --secret "<webhook_secret>" ^
    --api-key "<api_key>" ^
    --open-base http://127.0.0.1:8000 ^
    --output-dir ./partner_webhook_out

python ./app/client_scripts/test_partner_webhook.py --port 19090 --secret "whsec_jmJooJMuGPNJ310dq05j-de-U7ePYhP3" --api-key "pk_p1UKaodSy7Rw2-p3TlNgKDGD11zAGcruJLvXJdcGel8" --open-base http://127.0.0.1:8000 --output-dir ./partner_webhook_out
  管理台 Webhook URL 填: http://127.0.0.1:19090/webhook

签名算法:
  X-Signature = hex(HMAC-SHA256(webhook_secret, "{X-Timestamp}.{raw_body}"))
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import mimetypes
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urljoin, urlparse


def time_now() -> int:
    return int(datetime.now(timezone.utc).timestamp())


def verify_signature(secret: str, timestamp: str, body: bytes, signature: str) -> bool:
    if not secret:
        return True
    if not timestamp or not signature:
        return False
    expect = hmac.new(
        secret.encode("utf-8"),
        f"{timestamp}.".encode("utf-8") + body,
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expect, signature.strip().lower())


def _api_headers(api_key: str) -> Dict[str, str]:
    return {
        "X-API-Key": api_key,
        "Accept": "*/*",
        "User-Agent": "PartnerWebhookTest/1.0",
    }


def open_request(
    url: str,
    api_key: str,
    *,
    timeout: float = 30.0,
    method: str = "GET",
) -> Tuple[int, bytes, Dict[str, str], str]:
    """发起请求并跟随重定向（媒体接口会 302 到 MinIO）。"""
    current = url
    headers = _api_headers(api_key)
    for _ in range(8):
        req = urllib.request.Request(current, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                status = int(getattr(resp, "status", 200) or 200)
                data = resp.read()
                resp_headers = {k.lower(): v for k, v in resp.headers.items()}
                final_url = resp.geturl() or current
                return status, data, resp_headers, final_url
        except urllib.error.HTTPError as e:
            # 3xx 在部分环境下会进 HTTPError
            if e.code in (301, 302, 303, 307, 308):
                loc = e.headers.get("Location")
                if not loc:
                    raise
                current = urljoin(current, loc)
                # 跳到对象存储后通常不再带 API Key
                headers = {"User-Agent": "PartnerWebhookTest/1.0", "Accept": "*/*"}
                continue
            raise
    raise RuntimeError(f"重定向过多: {url}")


def fetch_alert_detail(
    base_url: str, api_key: str, alert_id: str, timeout: float = 15.0
) -> Dict[str, Any]:
    url = urljoin(base_url.rstrip("/") + "/", f"open/v1/alerts/{alert_id}")
    status, data, _, _ = open_request(url, api_key, timeout=timeout)
    if status >= 400:
        raise RuntimeError(f"详情 HTTP {status}: {data[:300]!r}")
    return json.loads(data.decode("utf-8") or "{}")


def _guess_ext(content_type: str, final_url: str, fallback: str) -> str:
    ctype = (content_type or "").split(";")[0].strip().lower()
    if ctype:
        ext = mimetypes.guess_extension(ctype)
        if ext == ".jpe":
            ext = ".jpg"
        if ext:
            return ext
    path = urlparse(final_url).path
    suf = Path(path).suffix
    if suf and len(suf) <= 5:
        return suf
    return fallback


def download_media(
    base_url: str,
    api_key: str,
    alert_id: str,
    kind: str,
    dest: Path,
    *,
    timeout: float = 60.0,
    retries: int = 1,
    retry_interval: float = 3.0,
) -> Optional[Path]:
    """
    kind: image | video
    优先走 Open API 媒体接口；失败时若详情里已有 url 也可由调用方兜底。
    """
    assert kind in {"image", "video"}
    api_path = f"open/v1/alerts/{alert_id}/media/{kind}"
    url = urljoin(base_url.rstrip("/") + "/", api_path)
    last_err: Optional[Exception] = None
    attempts = max(1, retries)
    for i in range(attempts):
        try:
            status, data, headers, final_url = open_request(
                url, api_key, timeout=timeout
            )
            if status >= 400:
                raise RuntimeError(f"HTTP {status}: {data[:200]!r}")
            if not data:
                raise RuntimeError("空响应体")
            ext = _guess_ext(
                headers.get("content-type", ""),
                final_url,
                ".jpg" if kind == "image" else ".mp4",
            )
            out = dest.with_suffix(ext)
            out.write_bytes(data)
            return out
        except Exception as e:
            last_err = e
            if i + 1 < attempts:
                print(f"  .. {kind} 未就绪，{retry_interval:.0f}s 后重试 ({i + 1}/{attempts})")
                time.sleep(retry_interval)
    if last_err:
        raise last_err
    return None


def download_by_url(url: str, dest: Path, *, timeout: float = 60.0) -> Path:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "PartnerWebhookTest/1.0", "Accept": "*/*"},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
        ctype = resp.headers.get("Content-Type", "")
        final = resp.geturl() or url
    ext = _guess_ext(ctype, final, dest.suffix or ".bin")
    out = dest.with_suffix(ext)
    out.write_bytes(data)
    return out


def pull_alert_assets(
    *,
    open_base: str,
    api_key: str,
    alert_id: str,
    output_dir: Path,
    video_retries: int,
    video_retry_interval: float,
    webhook_hint: Optional[Dict[str, Any]] = None,
) -> None:
    """拉取详情 JSON + 图片 + 视频，写入本地目录。"""
    hint = webhook_hint or {}
    safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in alert_id)[:80]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = output_dir / f"{stamp}_{safe_id}"
    folder.mkdir(parents=True, exist_ok=True)

    print(f"  → 拉取详情 JSON …")
    detail = fetch_alert_detail(open_base, api_key, alert_id)
    json_path = folder / "alert.json"
    json_path.write_text(
        json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  ✓ JSON 已保存: {json_path}")
    print(json.dumps(detail, ensure_ascii=False, indent=2))

    category = str(detail.get("category") or hint.get("category") or "").strip().lower()
    has_image_hint = bool(hint.get("has_image"))
    has_video_hint = bool(hint.get("has_video"))
    image_url = (detail.get("image_url") or "").strip()
    video_url = (detail.get("video_url") or "").strip()

    # 图片：有 URL / 通知声明有图 才下载
    if image_url or has_image_hint or category in {"alert", "event"}:
        try:
            print("  → 下载图片 …")
            img_path = download_media(
                open_base, api_key, alert_id, "image", folder / "alert_image"
            )
            if img_path:
                print(f"  ✓ 图片已保存: {img_path} ({img_path.stat().st_size} bytes)")
        except Exception as e:
            if image_url:
                try:
                    img_path = download_by_url(image_url, folder / "alert_image")
                    print(
                        f"  ✓ 图片已保存(直链): {img_path} ({img_path.stat().st_size} bytes)"
                    )
                except Exception as e2:
                    print(f"  !! 图片下载失败: {e} / 直链失败: {e2}")
            else:
                print(f"  - 暂无图片（跳过）: {e}")
    else:
        print("  - 通知未标记图片，跳过图片下载")

    # 视频：仅违规报警且任务开启证据视频时才会异步生成。
    # 事件（event / 画面人数 / 过线）默认无视频，不应轮询。
    expect_video = bool(video_url) or has_video_hint
    if not expect_video and category == "event":
        print("  - 类别为 event（事件），本系统不生成证据视频，跳过视频下载")
        print(f"  输出目录: {folder.resolve()}")
        return
    if not expect_video and category == "alert":
        # 报警也可能未开 alert_video_enabled，先试 1 次，404 则停
        print("  - 通知未标记 has_video，仅尝试 1 次视频（可能任务未开启报警视频）")
        video_retries = 1
    if not expect_video and category not in {"alert", "event"}:
        print("  - 无视频线索，跳过视频下载")
        print(f"  输出目录: {folder.resolve()}")
        return

    try:
        print(
            f"  → 下载视频（最多重试 {video_retries} 次，间隔 {video_retry_interval:.0f}s）…"
        )
        vid_path = download_media(
            open_base,
            api_key,
            alert_id,
            "video",
            folder / "alert_video",
            retries=video_retries,
            retry_interval=video_retry_interval,
        )
        if vid_path:
            print(f"  ✓ 视频已保存: {vid_path} ({vid_path.stat().st_size} bytes)")
    except Exception as e:
        if video_url:
            try:
                vid_path = download_by_url(video_url, folder / "alert_video")
                print(f"  ✓ 视频已保存(直链): {vid_path} ({vid_path.stat().st_size} bytes)")
            except Exception as e2:
                print(f"  !! 视频下载失败: {e} / 直链失败: {e2}")
        else:
            print(f"  - 暂无视频（未开启证据视频或尚未生成）: {e}")

    print(f"  输出目录: {folder.resolve()}")


def make_handler(
    *,
    secret: str,
    api_key: str,
    open_base: str,
    max_skew_sec: int,
    output_dir: Path,
    video_retries: int,
    video_retry_interval: float,
    fetch_async: bool,
):
    class PartnerWebhookHandler(BaseHTTPRequestHandler):
        server_version = "PartnerWebhookTest/1.0"

        def log_message(self, fmt: str, *args: Any) -> None:
            ts = datetime.now().strftime("%H:%M:%S")
            sys.stdout.write(f"[{ts}] {self.address_string()} - {fmt % args}\n")
            sys.stdout.flush()

        def _send_json(self, code: int, payload: Dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path.rstrip("/") in ("", "/", "/health"):
                self._send_json(200, {"status": "ok", "service": "partner-webhook-test"})
                return
            self._send_json(404, {"detail": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0].rstrip("/") or "/"
            if path not in ("/webhook", "/api/alerts/webhook"):
                self._send_json(404, {"detail": f"unknown path: {self.path}"})
                return

            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length > 0 else b""
            timestamp = self.headers.get("X-Timestamp") or ""
            signature = self.headers.get("X-Signature") or ""
            partner_code = self.headers.get("X-Partner-Code") or ""

            print("\n" + "=" * 60)
            print(f"收到 Webhook  {datetime.now().isoformat(timespec='seconds')}")
            print(f"  path           = {self.path}")
            print(f"  X-Partner-Code = {partner_code}")
            print(f"  X-Timestamp    = {timestamp}")
            print(
                f"  X-Signature    = {signature[:16]}..."
                if signature
                else "  X-Signature    = (无)"
            )

            if secret:
                if not verify_signature(secret, timestamp, body, signature):
                    print("  !! 签名校验失败")
                    self._send_json(401, {"detail": "invalid signature"})
                    return
                try:
                    skew = abs(int(time_now()) - int(timestamp))
                except ValueError:
                    skew = max_skew_sec + 1
                if skew > max_skew_sec:
                    print(f"  !! 时间戳偏差过大 skew={skew}s")
                    self._send_json(401, {"detail": "timestamp skew too large"})
                    return
                print("  签名校验通过")
            else:
                print("  （未配置 --secret，跳过签名校验）")

            try:
                payload = json.loads(body.decode("utf-8") or "{}")
            except json.JSONDecodeError:
                print("  !! JSON 解析失败 raw=", body[:200])
                self._send_json(400, {"detail": "invalid json"})
                return

            print("  通知正文:")
            print(json.dumps(payload, ensure_ascii=False, indent=2))

            alert_id = str(payload.get("alert_id") or "").strip()
            event_type = str(payload.get("event_type") or "")

            # 先快速回 200，再拉全量（更符合对方真实接入）
            self._send_json(200, {"ok": True, "received_alert_id": alert_id})

            if alert_id == "test_webhook_ping":
                print("  （测试推送 ping，跳过拉详情/媒体）")
                print("=" * 60 + "\n")
                sys.stdout.flush()
                return

            if not (api_key and open_base and alert_id):
                if not api_key:
                    print("  （未配置 --api-key，跳过拉详情/图片/视频）")
                print("=" * 60 + "\n")
                sys.stdout.flush()
                return

            def _job() -> None:
                try:
                    print(
                        f"  开始拉取全量资源 alert_id={alert_id} event_type={event_type}"
                    )
                    pull_alert_assets(
                        open_base=open_base,
                        api_key=api_key,
                        alert_id=alert_id,
                        output_dir=output_dir,
                        video_retries=video_retries,
                        video_retry_interval=video_retry_interval,
                        webhook_hint=payload,
                    )
                except Exception as e:
                    print(f"  !! 拉取全量失败: {e}")
                finally:
                    print("=" * 60 + "\n")
                    sys.stdout.flush()

            if fetch_async:
                threading.Thread(
                    target=_job, name=f"pull-{alert_id[:20]}", daemon=True
                ).start()
            else:
                _job()

    return PartnerWebhookHandler


def main() -> None:
    parser = argparse.ArgumentParser(description="第三方 Webhook 接收 + 拉详情/图片/视频")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址，默认 0.0.0.0")
    parser.add_argument("--port", type=int, default=19090, help="监听端口，默认 19090")
    parser.add_argument(
        "--secret",
        default=os.environ.get("PARTNER_WEBHOOK_SECRET", ""),
        help="Webhook 签名密钥；也可用环境变量 PARTNER_WEBHOOK_SECRET",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("PARTNER_API_KEY", ""),
        help="Open API Key；也可用环境变量 PARTNER_API_KEY",
    )
    parser.add_argument(
        "--open-base",
        default=os.environ.get("OPEN_API_BASE", "http://127.0.0.1:8000"),
        help="本项目服务根地址",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "partner_webhook_out"),
        help="详情 JSON / 图片 / 视频保存目录",
    )
    parser.add_argument(
        "--video-retries",
        type=int,
        default=8,
        help="视频未就绪时的重试次数，默认 8",
    )
    parser.add_argument(
        "--video-retry-interval",
        type=float,
        default=5.0,
        help="视频重试间隔秒，默认 5",
    )
    parser.add_argument(
        "--sync-fetch",
        action="store_true",
        help="同步拉全量（默认异步：先回 200 再拉）",
    )
    parser.add_argument(
        "--max-skew",
        type=int,
        default=300,
        help="允许的时间戳偏差秒数，默认 300",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    handler = make_handler(
        secret=args.secret.strip(),
        api_key=args.api_key.strip(),
        open_base=args.open_base.strip(),
        max_skew_sec=args.max_skew,
        output_dir=output_dir,
        video_retries=max(1, args.video_retries),
        video_retry_interval=max(0.5, args.video_retry_interval),
        fetch_async=not args.sync_fetch,
    )
    server = ThreadingHTTPServer((args.host, args.port), handler)
    local_url = f"http://127.0.0.1:{args.port}/webhook"
    print("第三方 Webhook 测试服务已启动")
    print(f"  监听: {args.host}:{args.port}")
    print(f"  Webhook URL: {local_url}")
    print(f"  输出目录: {output_dir.resolve()}")
    print(f"  签名校验: {'开启' if args.secret.strip() else '关闭（建议传 --secret）'}")
    print(
        f"  拉全量: {'开启 → ' + args.open_base if args.api_key.strip() else '关闭（需 --api-key）'}"
    )
    print(
        f"  拉取模式: {'同步' if args.sync_fetch else '异步（先回 200）'}；"
        f"视频重试 {args.video_retries}×{args.video_retry_interval}s"
    )
    print("按 Ctrl+C 结束\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
        server.server_close()


if __name__ == "__main__":
    main()
