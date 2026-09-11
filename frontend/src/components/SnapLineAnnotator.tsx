import { useEffect, useRef, useState, MouseEvent } from "react";
import { getToken } from "../auth";

export type Pt = { x: number; y: number };

export type LineItem = {
  line: [Pt, Pt];
  side_point: Pt;
};

type Tool =
  | "count_seg" // 过线：先两点成线，再一点为 enter_point
  | "bypass_seg"
  | "line"
  | "point"
  | "rect"
  | "polygon";

type Shape =
  | { kind: "line"; a: Pt; b: Pt }
  | { kind: "point"; p: Pt }
  | { kind: "rect"; a: Pt; b: Pt }
  | { kind: "polygon"; points: Pt[] };

type Props = {
  title: string;
  /** count_line 用 enter_point；bypass_line 用 bypass_point */
  target: "count_line" | "bypass_line";
  cameras: any[];
  /** 默认选中的摄像头 id（任务表单里已选的摄像头） */
  initialCameraId?: number | "";
  initialJson?: string;
  onClose: () => void;
  onConfirm: (jsonText: string) => void;
};

function parseInitial(
  text: string | undefined,
  target: "count_line" | "bypass_line"
): LineItem[] {
  if (!text?.trim()) return [];
  try {
    const obj = JSON.parse(text);
    const lines = obj?.lines;
    if (!Array.isArray(lines)) return [];
    const key = target === "count_line" ? "enter_point" : "bypass_point";
    const out: LineItem[] = [];
    for (const it of lines) {
      const line = it?.line;
      const side = it?.[key];
      if (
        Array.isArray(line) &&
        line.length >= 2 &&
        side &&
        typeof side.x === "number" &&
        typeof side.y === "number"
      ) {
        out.push({
          line: [
            { x: Math.round(line[0].x), y: Math.round(line[0].y) },
            { x: Math.round(line[1].x), y: Math.round(line[1].y) },
          ],
          side_point: { x: Math.round(side.x), y: Math.round(side.y) },
        });
      }
    }
    return out;
  } catch {
    return [];
  }
}

function toJson(
  items: LineItem[],
  target: "count_line" | "bypass_line",
  imageSize?: { w: number; h: number }
): string {
  const key = target === "count_line" ? "enter_point" : "bypass_point";
  const payload: Record<string, unknown> = {
    lines: items.map((it) => ({
      line: it.line,
      [key]: it.side_point,
    })),
  };
  // 写入截图分辨率，推流帧尺寸不一致时技能侧会按比例缩放
  if (imageSize && imageSize.w > 0 && imageSize.h > 0) {
    payload.image_width = imageSize.w;
    payload.image_height = imageSize.h;
  }
  return JSON.stringify(payload, null, 2);
}

function rectPoints(a: Pt, b: Pt): Pt[] {
  const x1 = Math.min(a.x, b.x);
  const y1 = Math.min(a.y, b.y);
  const x2 = Math.max(a.x, b.x);
  const y2 = Math.max(a.y, b.y);
  return [
    { x: x1, y: y1 },
    { x: x2, y: y1 },
    { x: x2, y: y2 },
    { x: x1, y: y2 },
  ];
}

export default function SnapLineAnnotator({
  title,
  target,
  cameras,
  initialCameraId,
  initialJson,
  onClose,
  onConfirm,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);
  const [cameraId, setCameraId] = useState<number | "">(
    initialCameraId || cameras[0]?.id || ""
  );
  const [snapUrl, setSnapUrl] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [tool, setTool] = useState<Tool>(
    target === "count_line" ? "count_seg" : "bypass_seg"
  );
  const [items, setItems] = useState<LineItem[]>(() =>
    parseInitial(initialJson, target)
  );
  const [shapes, setShapes] = useState<Shape[]>([]);
  const [draftPts, setDraftPts] = useState<Pt[]>([]);
  const [imgSize, setImgSize] = useState({ w: 0, h: 0 });

  const sideLabel = target === "count_line" ? "enter_point" : "bypass_point";

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const loadSnap = async () => {
    setError("");
    setLoading(true);
    try {
      const q = new URLSearchParams();
      if (cameraId !== "") q.set("camera_id", String(cameraId));
      else {
        setError("请选择摄像头");
        setLoading(false);
        return;
      }
      q.set("timeout_sec", "15");
      q.set("expire_sec", "30");
      const token = getToken();
      const res = await fetch(`/api/v1/zlm/snap?${q}`, {
        headers: token ? { Authorization: `Bearer ${token}` } : undefined,
      });
      if (!res.ok) {
        let detail = res.statusText;
        try {
          const body = await res.json();
          detail = body.detail || JSON.stringify(body);
        } catch {
          /* ignore */
        }
        throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
      }
      const blob = await res.blob();
      if (snapUrl) URL.revokeObjectURL(snapUrl);
      const obj = URL.createObjectURL(blob);
      setSnapUrl(obj);
      setDraftPts([]);
    } catch (e: any) {
      setError(e.message || String(e));
    } finally {
      setLoading(false);
    }
  };

  const redraw = () => {
    const canvas = canvasRef.current;
    const img = imgRef.current;
    if (!canvas || !img || !img.naturalWidth) return;
    const w = img.naturalWidth;
    const h = img.naturalHeight;
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, w, h);
    ctx.drawImage(img, 0, 0);

    const drawPt = (p: Pt, color: string, r = 5) => {
      ctx.beginPath();
      ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 1;
      ctx.stroke();
    };
    const drawSeg = (a: Pt, b: Pt, color: string) => {
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.stroke();
      drawPt(a, color, 4);
      drawPt(b, color, 4);
    };

    items.forEach((it, idx) => {
      drawSeg(it.line[0], it.line[1], "#3d9b6e");
      drawPt(it.side_point, "#c4a35a", 6);
      ctx.fillStyle = "#e8eee9";
      ctx.font = "14px sans-serif";
      ctx.fillText(`#${idx + 1}`, it.line[0].x + 8, it.line[0].y - 8);
    });

    shapes.forEach((s) => {
      if (s.kind === "line") drawSeg(s.a, s.b, "#5b9bd5");
      if (s.kind === "point") drawPt(s.p, "#5b9bd5");
      if (s.kind === "rect") {
        const pts = rectPoints(s.a, s.b);
        ctx.beginPath();
        ctx.moveTo(pts[0].x, pts[0].y);
        pts.slice(1).forEach((p) => ctx.lineTo(p.x, p.y));
        ctx.closePath();
        ctx.strokeStyle = "#9b7edc";
        ctx.lineWidth = 2;
        ctx.stroke();
        pts.forEach((p) => drawPt(p, "#9b7edc", 3));
      }
      if (s.kind === "polygon") {
        if (s.points.length) {
          ctx.beginPath();
          ctx.moveTo(s.points[0].x, s.points[0].y);
          s.points.slice(1).forEach((p) => ctx.lineTo(p.x, p.y));
          if (s.points.length > 2) ctx.closePath();
          ctx.strokeStyle = "#e07a5f";
          ctx.lineWidth = 2;
          ctx.stroke();
          s.points.forEach((p) => drawPt(p, "#e07a5f", 3));
        }
      }
    });

    // draft
    draftPts.forEach((p, i) => {
      drawPt(p, "#fff", 4);
      if (i > 0) {
        ctx.beginPath();
        ctx.moveTo(draftPts[i - 1].x, draftPts[i - 1].y);
        ctx.lineTo(p.x, p.y);
        ctx.setLineDash([6, 4]);
        ctx.strokeStyle = "#fff";
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.setLineDash([]);
      }
    });
  };

  useEffect(() => {
    redraw();
  }, [items, shapes, draftPts, snapUrl, imgSize]);

  const canvasToImagePt = (e: MouseEvent<HTMLCanvasElement>): Pt | null => {
    const canvas = canvasRef.current;
    if (!canvas || !imgSize.w) return null;
    const rect = canvas.getBoundingClientRect();
    const scaleX = imgSize.w / rect.width;
    const scaleY = imgSize.h / rect.height;
    return {
      x: Math.round((e.clientX - rect.left) * scaleX),
      y: Math.round((e.clientY - rect.top) * scaleY),
    };
  };

  const onCanvasClick = (e: MouseEvent<HTMLCanvasElement>) => {
    const p = canvasToImagePt(e);
    if (!p) return;

    if (tool === "count_seg" || tool === "bypass_seg") {
      const next = [...draftPts, p];
      if (next.length < 3) {
        setDraftPts(next);
        return;
      }
      // 两点线段 + 侧向点
      setItems((prev) => [
        ...prev,
        { line: [next[0], next[1]], side_point: next[2] },
      ]);
      setDraftPts([]);
      return;
    }

    if (tool === "line") {
      const next = [...draftPts, p];
      if (next.length < 2) {
        setDraftPts(next);
        return;
      }
      setShapes((prev) => [...prev, { kind: "line", a: next[0], b: next[1] }]);
      setDraftPts([]);
      return;
    }

    if (tool === "point") {
      setShapes((prev) => [...prev, { kind: "point", p }]);
      return;
    }

    if (tool === "rect") {
      const next = [...draftPts, p];
      if (next.length < 2) {
        setDraftPts(next);
        return;
      }
      setShapes((prev) => [...prev, { kind: "rect", a: next[0], b: next[1] }]);
      setDraftPts([]);
      return;
    }

    if (tool === "polygon") {
      setDraftPts((prev) => [...prev, p]);
    }
  };

  const onCanvasDblClick = (e: MouseEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    if (tool !== "polygon") return;
    if (draftPts.length < 3) {
      setError("多边形至少需要 3 个点");
      return;
    }
    setShapes((prev) => [...prev, { kind: "polygon", points: [...draftPts] }]);
    setDraftPts([]);
  };

  const hint = () => {
    if (tool === "count_seg" || tool === "bypass_seg") {
      if (draftPts.length === 0) return "点击第 1 个端点";
      if (draftPts.length === 1) return "点击第 2 个端点完成线段";
      return `点击 ${sideLabel}（进入/参考侧点）`;
    }
    if (tool === "line")
      return draftPts.length ? "点击第 2 点完成线段" : "点击第 1 点";
    if (tool === "point") return "点击添加点";
    if (tool === "rect")
      return draftPts.length ? "点击对角完成矩形" : "点击矩形一角";
    if (tool === "polygon")
      return "依次点击顶点，双击结束多边形";
    return "";
  };

  const jsonPreview = toJson(items, target, imgSize);

  return (
    <div className="modal-backdrop" onClick={onClose} role="presentation">
      <div
        className="modal-panel annotator-panel"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
      >
        <div className="modal-head">
          <div>
            <h2>{title}</h2>
            <p className="muted">
              坐标为截图像素（会写入 image_width/height）；推流分辨率不同时自动缩放。
              过线需「线段两点 + {sideLabel}」。
              {imgSize.w > 0 ? ` 当前截图 ${imgSize.w}×${imgSize.h}` : ""}
            </p>
          </div>
          <button className="btn" type="button" onClick={onClose}>
            关闭
          </button>
        </div>

        <div className="annotator-toolbar">
          <label>
            摄像头
            <select
              value={cameraId}
              onChange={(e) =>
                setCameraId(e.target.value ? Number(e.target.value) : "")
              }
            >
              {!cameras.length && <option value="">无摄像头</option>}
              {cameras.map((c) => (
                <option key={c.id} value={c.id}>
                  #{c.id} {c.name}
                </option>
              ))}
            </select>
          </label>
          <button className="btn primary" type="button" onClick={loadSnap} disabled={loading}>
            {loading ? "截图中…" : "拉取 ZLM 截图"}
          </button>
          <span className="muted">{hint()}</span>
        </div>

        <div className="annotator-tools">
          {(
            [
              [target === "count_line" ? "count_seg" : "bypass_seg", "过线/绕行(线+点)"],
              ["line", "线段"],
              ["point", "点"],
              ["rect", "矩形"],
              ["polygon", "多边形"],
            ] as [Tool, string][]
          ).map(([t, label]) => (
            <button
              key={t}
              type="button"
              className={`btn ${tool === t ? "primary" : ""}`}
              onClick={() => {
                setTool(t);
                setDraftPts([]);
              }}
            >
              {label}
            </button>
          ))}
          <button
            type="button"
            className="btn"
            onClick={() => {
              setDraftPts([]);
            }}
          >
            取消当前绘制
          </button>
          <button
            type="button"
            className="btn danger"
            onClick={() => {
              setItems([]);
              setShapes([]);
              setDraftPts([]);
            }}
          >
            清空全部
          </button>
          {items.length > 0 && (
            <button
              type="button"
              className="btn"
              onClick={() => setItems((prev) => prev.slice(0, -1))}
            >
              撤销上一段过线
            </button>
          )}
        </div>

        {error && <p className="error">{error}</p>}

        <div className="annotator-body">
          <div className="annotator-canvas-wrap">
            {!snapUrl && (
              <p className="muted" style={{ padding: 24 }}>
                请选择摄像头并点击「拉取 ZLM 截图」
              </p>
            )}
            {snapUrl && (
              <>
                <img
                  ref={imgRef}
                  src={snapUrl}
                  alt="snap"
                  style={{ display: "none" }}
                  onLoad={(e) => {
                    const img = e.currentTarget;
                    setImgSize({ w: img.naturalWidth, h: img.naturalHeight });
                    requestAnimationFrame(redraw);
                  }}
                />
                <canvas
                  ref={canvasRef}
                  className="annotator-canvas"
                  onClick={onCanvasClick}
                  onDoubleClick={onCanvasDblClick}
                />
                <p className="muted mono" style={{ marginTop: 6 }}>
                  原图 {imgSize.w}×{imgSize.h}
                </p>
              </>
            )}
          </div>
          <div className="annotator-side">
            <h3 className="section-title">
              {target} JSON（{items.length} 段）
            </h3>
            <textarea
              className="annotator-json"
              readOnly
              value={jsonPreview}
            />
            {shapes.length > 0 && (
              <>
                <h3 className="section-title">其它图形点位</h3>
                <pre className="annotator-shapes mono">
                  {JSON.stringify(
                    shapes.map((s) => {
                      if (s.kind === "rect")
                        return { kind: "rect", points: rectPoints(s.a, s.b) };
                      if (s.kind === "line")
                        return { kind: "line", points: [s.a, s.b] };
                      if (s.kind === "point")
                        return { kind: "point", points: [s.p] };
                      return { kind: "polygon", points: s.points };
                    }),
                    null,
                    2
                  )}
                </pre>
              </>
            )}
            <div className="toolbar" style={{ marginTop: 12 }}>
              <button
                className="btn primary"
                type="button"
                disabled={!items.length}
                onClick={() => onConfirm(jsonPreview)}
              >
                应用到表单
              </button>
              <button className="btn ghost" type="button" onClick={onClose}>
                取消
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
