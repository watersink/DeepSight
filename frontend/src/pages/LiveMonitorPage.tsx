import { useCallback, useEffect, useMemo, useState } from "react";
import { api } from "../api";
import Pager from "../components/Pager";
import StreamPlayer from "../components/StreamPlayer";

type Layout = 1 | 4 | 9 | 16;

const LAYOUTS: Layout[] = [1, 4, 9, 16];
const STORAGE_LAYOUT = "live_monitor_layout";
const STORAGE_SLOTS = "live_monitor_slots";

function loadLayout(): Layout {
  try {
    const n = Number(localStorage.getItem(STORAGE_LAYOUT));
    if (n === 1 || n === 4 || n === 9 || n === 16) return n;
  } catch {
    /* ignore */
  }
  return 4;
}

/** 读取完整槽位列表（可超过当前分屏数，供翻页） */
function loadSlots(): (number | null)[] {
  try {
    const raw = localStorage.getItem(STORAGE_SLOTS);
    if (!raw) return [];
    const arr = JSON.parse(raw);
    if (!Array.isArray(arr)) return [];
    return arr.map((v) =>
      typeof v === "number" && Number.isFinite(v) ? v : null
    );
  } catch {
    return [];
  }
}

function layoutCols(layout: Layout): number {
  if (layout === 1) return 1;
  if (layout === 4) return 2;
  if (layout === 9) return 3;
  return 4;
}

export default function LiveMonitorPage() {
  const [layout, setLayout] = useState<Layout>(loadLayout);
  const [slots, setSlots] = useState<(number | null)[]>(loadSlots);
  const [page, setPage] = useState(1);
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

  /** 翻页总数：至少一页分屏格数；槽位更多时按槽位长度分页 */
  const pagerTotal = Math.max(slots.length, layout);
  const totalPages = Math.max(1, Math.ceil(pagerTotal / layout) || 1);
  const safePage = Math.min(Math.max(1, page), totalPages);
  const pageStart = (safePage - 1) * layout;

  const pageSlots = useMemo(
    () =>
      Array.from({ length: layout }, (_, i) => slots[pageStart + i] ?? null),
    [layout, slots, pageStart]
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

  useEffect(() => {
    if (page !== safePage) setPage(safePage);
  }, [page, safePage]);

  const changeLayout = (next: Layout) => {
    setLayout(next);
    setPage(1);
  };

  const setSlotTask = (indexOnPage: number, taskId: number | null) => {
    const globalIdx = pageStart + indexOnPage;
    setSlots((prev) => {
      const next = [...prev];
      while (next.length <= globalIdx) next.push(null);
      next[globalIdx] = taskId;
      return next;
    });
  };

  const autoFill = () => {
    const candidates = playableTasks.filter((t) => t.flv_url);
    const running = candidates.filter((t) => t.runtime_status === "running");
    const pool = running.length ? running : candidates;
    setSlots(pool.map((t) => t.id as number));
    setPage(1);
  };

  const cols = layoutCols(layout);

  return (
    <div className="page-shell live-monitor-page">
      <div className="page-head">
        <div>
          <h1>实时展示</h1>
          <p>
            分屏预览任务配置中的识别输出流（需任务已启动并产生 FLV）。
            视频多于当前分屏时，可用底部分页切换。
          </p>
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
        className={`live-grid live-grid-${layout}`}
        style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` }}
      >
        {pageSlots.map((taskId, idx) => {
          const globalIdx = pageStart + idx;
          const task = taskId != null ? taskById.get(taskId) : null;
          const flv = task?.flv_url || null;
          return (
            <section key={`${layout}-p${safePage}-${idx}`} className="live-cell">
              <div className="live-cell-head">
                <span className="live-cell-index">#{globalIdx + 1}</span>
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

      <Pager
        page={safePage}
        pageSize={layout}
        total={pagerTotal}
        onChange={setPage}
      />
    </div>
  );
}
