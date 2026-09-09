import { useCallback, useEffect, useMemo, useState } from "react";
import { api } from "../api";
import StreamPlayer from "../components/StreamPlayer";

type Layout = 1 | 4 | 9;

const LAYOUTS: Layout[] = [1, 4, 9];
const STORAGE_LAYOUT = "live_monitor_layout";
const STORAGE_SLOTS = "live_monitor_slots";

function loadLayout(): Layout {
  try {
    const n = Number(localStorage.getItem(STORAGE_LAYOUT));
    if (n === 1 || n === 4 || n === 9) return n;
  } catch {
    /* ignore */
  }
  return 4;
}

function loadSlots(size: number): (number | null)[] {
  try {
    const raw = localStorage.getItem(STORAGE_SLOTS);
    if (!raw) return Array.from({ length: size }, () => null);
    const arr = JSON.parse(raw);
    if (!Array.isArray(arr)) return Array.from({ length: size }, () => null);
    const next = Array.from({ length: size }, (_, i) => {
      const v = arr[i];
      return typeof v === "number" && Number.isFinite(v) ? v : null;
    });
    return next;
  } catch {
    return Array.from({ length: size }, () => null);
  }
}

export default function LiveMonitorPage() {
  const [layout, setLayout] = useState<Layout>(loadLayout);
  const [slots, setSlots] = useState<(number | null)[]>(() =>
    loadSlots(loadLayout())
  );
  const [tasks, setTasks] = useState<any[]>([]);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const taskById = useMemo(() => {
    const m = new Map<number, any>();
    for (const t of tasks) m.set(t.id, t);
    return m;
  }, [tasks]);

  const playableTasks = useMemo(
    () =>
      tasks.filter(
        (t) => t.flv_url && (t.runtime_status === "running" || t.enabled !== false)
      ),
    [tasks]
  );

  const load = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const data = await api.getTasks({ page: 1, pageSize: 200 });
      setTasks(data.items || []);
    } catch (e: any) {
      setError(e.message || String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
    const timer = setInterval(load, 10000);
    return () => clearInterval(timer);
  }, [load]);

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_LAYOUT, String(layout));
      localStorage.setItem(STORAGE_SLOTS, JSON.stringify(slots));
    } catch {
      /* ignore */
    }
  }, [layout, slots]);

  const changeLayout = (next: Layout) => {
    setLayout(next);
    setSlots((prev) => {
      const out = Array.from({ length: next }, (_, i) => prev[i] ?? null);
      return out;
    });
  };

  const setSlotTask = (index: number, taskId: number | null) => {
    setSlots((prev) => {
      const next = [...prev];
      next[index] = taskId;
      return next;
    });
  };

  const autoFill = () => {
    const candidates = playableTasks.filter((t) => t.flv_url);
    const running = candidates.filter((t) => t.runtime_status === "running");
    const pool = (running.length ? running : candidates).slice(0, layout);
    setSlots(
      Array.from({ length: layout }, (_, i) => (pool[i] ? pool[i].id : null))
    );
  };

  const cols = layout === 1 ? 1 : layout === 4 ? 2 : 3;

  return (
    <div className="page-shell live-monitor-page">
      <div className="page-head">
        <div>
          <h1>实时展示</h1>
          <p>分屏预览任务配置中的识别输出流（需任务已启动并产生 FLV）。</p>
        </div>
        <div className="toolbar">
          <div className="view-toggle" role="group" aria-label="分屏方式">
            {LAYOUTS.map((n) => (
              <button
                key={n}
                type="button"
                className={`btn view-toggle-btn ${layout === n ? "active" : ""}`}
                aria-pressed={layout === n}
                onClick={() => changeLayout(n)}
              >
                {n} 分屏
              </button>
            ))}
          </div>
          <button className="btn" type="button" onClick={autoFill}>
            自动填充
          </button>
          <button className="btn" type="button" onClick={load} disabled={loading}>
            {loading ? "刷新中…" : "刷新任务"}
          </button>
        </div>
      </div>

      {error && <p className="error">{error}</p>}

      <div
        className="live-grid"
        style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` }}
      >
        {slots.map((taskId, idx) => {
          const task = taskId != null ? taskById.get(taskId) : null;
          const flv = task?.flv_url || null;
          return (
            <section key={`${layout}-${idx}`} className="live-cell">
              <div className="live-cell-head">
                <span className="live-cell-index">#{idx + 1}</span>
                <select
                  value={taskId ?? ""}
                  onChange={(e) => {
                    const v = e.target.value;
                    setSlotTask(idx, v ? Number(v) : null);
                  }}
                >
                  <option value="">选择任务…</option>
                  {playableTasks.map((t) => (
                    <option key={t.id} value={t.id}>
                      {t.name}
                      {t.runtime_status === "running" ? " · 运行中" : ""}
                      {!t.flv_url ? " · 无FLV" : ""}
                    </option>
                  ))}
                  {taskId != null && !playableTasks.some((t) => t.id === taskId) && (
                    <option value={taskId}>
                      {task?.name || `任务#${taskId}`}（当前不可播）
                    </option>
                  )}
                </select>
              </div>
              <div className="live-cell-body">
                {flv ? (
                  <StreamPlayer flvUrl={flv} compact />
                ) : (
                  <div className="stream-player empty">
                    <p className="muted">
                      {taskId
                        ? "该任务暂无 FLV，请先在任务配置中启动"
                        : "请选择已启动的识别任务"}
                    </p>
                  </div>
                )}
              </div>
              {task && (
                <div className="live-cell-foot muted mono">
                  {task.camera_name || `cam#${task.camera_id}`} ·{" "}
                  {task.skill_name || "-"} · {task.scene_id}
                </div>
              )}
            </section>
          );
        })}
      </div>
    </div>
  );
}
