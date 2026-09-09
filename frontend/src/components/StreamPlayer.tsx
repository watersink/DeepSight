import { useEffect, useRef, useState } from "react";
import {
  decoderUrl,
  loadJessibucaScript,
  toPlayUrl,
  type JessibucaPlayer,
} from "./jessibuca";

type Props = {
  flvUrl?: string | null;
  /** 分屏内隐藏多余控件，降低开销 */
  compact?: boolean;
};

export default function StreamPlayer({ flvUrl, compact = false }: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const playerRef = useRef<JessibucaPlayer | null>(null);
  const [error, setError] = useState("");
  const [ready, setReady] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setError("");
    setReady(false);

    if (!flvUrl) {
      setError("");
      return;
    }
    if (!containerRef.current) return;

    const url = toPlayUrl(flvUrl);

    (async () => {
      try {
        const Jessibuca = await loadJessibucaScript();
        if (cancelled || !containerRef.current) return;

        containerRef.current.innerHTML = "";

        const player = new Jessibuca({
          container: containerRef.current,
          videoBuffer: compact ? 0.8 : 1,
          isResize: true,
          loadingText: "加载中...",
          debug: false,
          showBandwidth: !compact,
          hasAudio: false,
          hasVideo: true,
          isFlv: true,
          operateBtns: {
            fullscreen: !compact,
            screenshot: false,
            play: !compact,
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
            setError(
              `播放失败: ${typeof err === "string" ? err : JSON.stringify(err)}`
            );
          }
        });
        player.on("timeout", () => {
          if (!cancelled) setError("播放超时，请确认任务已启动且流在线");
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
  }, [flvUrl, compact]);

  if (!flvUrl) {
    return (
      <div className="stream-player empty">
        <p className="muted">未选择任务或无可用 FLV</p>
      </div>
    );
  }

  return (
    <div className={`stream-player ${compact ? "compact" : ""}`}>
      <div ref={containerRef} className="jessibuca-container fill" />
      {error && <p className="stream-player-msg error">{error}</p>}
      {!error && !ready && (
        <p className="stream-player-msg muted">连接中…</p>
      )}
    </div>
  );
}
