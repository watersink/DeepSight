import { useEffect, useRef, useState } from "react";

type Props = {
  title: string;
  flvUrl?: string | null;
  onClose: () => void;
};

type JessibucaPlayer = {
  play: (url: string) => void;
  pause?: () => void;
  destroy: () => void;
  on: (event: string, callback: (value?: any) => void) => void;
  setVolume?: (v: number) => void;
  mute?: () => void;
};

type JessibucaConstructor = new (options: Record<string, any>) => JessibucaPlayer;

declare global {
  interface Window {
    Jessibuca?: JessibucaConstructor;
    jessibuca?: JessibucaConstructor;
  }
}

const JESSIBUCA_SCRIPT = "/jessibuca/jessibuca.js";

function resolveJessibuca(): JessibucaConstructor | undefined {
  return window.Jessibuca || window.jessibuca;
}

function decoderUrl(): string {
  // Worker 内相对路径易错，用绝对地址保证能拉到 decoder.wasm
  return `${window.location.origin}/jessibuca/decoder.js`;
}

/** Jessibuca 对 WS-FLV 更稳；ZLM 同一路径同时支持 http/ws */
function toPlayUrl(httpFlv: string): string {
  try {
    const u = new URL(httpFlv);
    if (u.protocol === "http:") u.protocol = "ws:";
    else if (u.protocol === "https:") u.protocol = "wss:";
    return u.toString();
  } catch {
    return httpFlv;
  }
}

function loadJessibucaScript(): Promise<JessibucaConstructor> {
  const existingCtor = resolveJessibuca();
  if (existingCtor) {
    return Promise.resolve(existingCtor);
  }
  return new Promise((resolve, reject) => {
    const existing = document.querySelector<HTMLScriptElement>(
      'script[data-jessibuca="1"]'
    );
    if (existing) {
      existing.addEventListener("load", () => {
        const ctor = resolveJessibuca();
        if (ctor) resolve(ctor);
        else reject(new Error("Jessibuca 加载后不可用"));
      });
      existing.addEventListener("error", () =>
        reject(new Error("Jessibuca 脚本加载失败"))
      );
      return;
    }
    const script = document.createElement("script");
    script.src = JESSIBUCA_SCRIPT;
    script.async = true;
    script.dataset.jessibuca = "1";
    script.onload = () => {
      const ctor = resolveJessibuca();
      if (ctor) resolve(ctor);
      else reject(new Error("Jessibuca 加载后不可用"));
    };
    script.onerror = () =>
      reject(new Error("Jessibuca 脚本加载失败，请确认 public/jessibuca 资源存在"));
    document.head.appendChild(script);
  });
}

export default function StreamPlayerModal({ title, flvUrl, onClose }: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const playerRef = useRef<JessibucaPlayer | null>(null);
  const [error, setError] = useState("");
  const [ready, setReady] = useState(false);
  const [playUrl, setPlayUrl] = useState("");

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  useEffect(() => {
    let cancelled = false;
    setError("");
    setReady(false);
    setPlayUrl("");

    if (!flvUrl) {
      setError("无可播放地址（需 RTMP/RTSP 流，由 ZLM 转 HTTP/WS-FLV）");
      return;
    }
    const container = containerRef.current;
    if (!container) return;

    const url = toPlayUrl(flvUrl);
    setPlayUrl(url);

    (async () => {
      try {
        const Jessibuca = await loadJessibucaScript();
        if (cancelled || !containerRef.current) return;

        // Jessibuca 会往 container 内写入 DOM，先清空避免重复挂载
        containerRef.current.innerHTML = "";

        const player = new Jessibuca({
          container: containerRef.current,
          // 秒；内部会 *1000。过小易卡顿/黑闪，监控流用 1s 更稳
          videoBuffer: 1,
          isResize: true,
          loadingText: "加载中...",
          debug: false,
          showBandwidth: true,
          // 监控流常无音轨；hasAudio 默认 true 会导致 demux 卡住甚至黑屏
          hasAudio: false,
          hasVideo: true,
          isFlv: true,
          operateBtns: {
            fullscreen: true,
            screenshot: true,
            play: true,
            audio: false,
          },
          decoder: decoderUrl(),
          forceNoOffscreen: true,
          isNotMute: false,
          timeout: 15,
          heartTimeout: 15,
          heartTimeoutReplay: true,
          heartTimeoutReplayTimes: 3,
        });

        playerRef.current = player;

        player.on("error", (err) => {
          if (!cancelled) {
            setError(`播放失败: ${typeof err === "string" ? err : JSON.stringify(err)}`);
          }
        });
        player.on("timeout", () => {
          if (!cancelled) setError("播放超时，请确认流是否在线、本机能否访问 ZLM");
        });
        player.on("play", () => {
          if (!cancelled) {
            setReady(true);
            setError("");
          }
        });
        player.on("start", () => {
          if (!cancelled) {
            setReady(true);
            setError("");
          }
        });

        player.mute?.();
        player.play(url);
      } catch (e: any) {
        if (!cancelled) setError(e?.message || String(e));
      }
    })();

    return () => {
      cancelled = true;
      const player = playerRef.current;
      if (player) {
        try {
          player.pause?.();
          player.destroy();
        } catch {
          /* ignore */
        }
        playerRef.current = null;
      }
      if (containerRef.current) {
        containerRef.current.innerHTML = "";
      }
    };
  }, [flvUrl]);

  return (
    <div className="modal-backdrop" onClick={onClose} role="presentation">
      <div
        className="modal-panel"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-label={title}
      >
        <div className="modal-head">
          <div>
            <h2>{title}</h2>
            {(playUrl || flvUrl) && (
              <p className="mono muted">{playUrl || flvUrl}</p>
            )}
          </div>
          <button className="btn" type="button" onClick={onClose}>
            关闭
          </button>
        </div>
        <div className="modal-body">
          <div ref={containerRef} className="jessibuca-container" />
          {error && <p className="error">{error}</p>}
          {!error && (
            <p className="muted" style={{ marginTop: 8 }}>
              Jessibuca 播放 ZLMediaKit WS-FLV
              {ready ? "（已开始播放）" : "；流离线或跨网段不通时可能黑屏"}。
              支持 H.264 / H.265（WASM 软解）。
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
