import { useEffect, useState } from "react";
import { api } from "../api";
import Pager from "../components/Pager";

const PAGE_SIZE = 10;

function EvidenceIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true">
      <path
        fill="currentColor"
        d="M4 5h3.2l1.3-2h7l1.3 2H20a2 2 0 0 1 2 2v11a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2zm8 3.5A4.5 4.5 0 1 0 16.5 13 4.5 4.5 0 0 0 12 8.5zm0 2A2.5 2.5 0 1 1 9.5 13 2.5 2.5 0 0 1 12 10.5z"
      />
    </svg>
  );
}

function typeLabel(a: any): string {
  const codes = (a.recognition_types || []).filter(Boolean);
  if (codes.length) return codes.join(",");
  if (a.skill_name === "person_presence_detector26") return "画面人数";
  return "-";
}

export default function EventsPage() {
  const [items, setItems] = useState<any[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [status, setStatus] = useState("");
  const [sceneId, setSceneId] = useState("");
  const [skillName, setSkillName] = useState("");
  const [recognitionType, setRecognitionType] = useState("");
  const [timeFrom, setTimeFrom] = useState("");
  const [timeTo, setTimeTo] = useState("");
  const [error, setError] = useState("");
  const [selected, setSelected] = useState<any | null>(null);
  const [preview, setPreview] = useState<any | null>(null);

  const filterKey = `${status}|${sceneId}|${skillName}|${recognitionType}|${timeFrom}|${timeTo}`;

  const load = async (pageNo = page) => {
    setError("");
    try {
      const isPresence = recognitionType === "presence";
      const data = await api.getEvents({
        status: status || undefined,
        sceneId: sceneId.trim() || undefined,
        skillName: isPresence
          ? "person_presence_detector26"
          : skillName.trim() || undefined,
        recognitionType:
          !isPresence && recognitionType ? recognitionType : undefined,
        timeFrom: timeFrom || undefined,
        timeTo: timeTo || undefined,
        page: pageNo,
        pageSize: PAGE_SIZE,
      });
      setItems(data.items || []);
      setTotal(data.meta?.total ?? (data.items || []).length);
    } catch (e: any) {
      setError(e.message || String(e));
    }
  };

  useEffect(() => {
    setPage(1);
  }, [filterKey]);

  useEffect(() => {
    load(page);
    const timer = setInterval(() => load(page), 8000);
    return () => clearInterval(timer);
  }, [filterKey, page]);

  useEffect(() => {
    if (!preview) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setPreview(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [preview]);

  return (
    <div className="page-shell">
      <div className="page-head">
        <div>
          <h1>事件管理</h1>
          <p>
            正常识别结果：过线计数（01/02）、画面人数等；违规（04–07）请到「报警管理」。
          </p>
        </div>
        <div className="toolbar">
          <input
            type="text"
            placeholder="场景 ID"
            value={sceneId}
            onChange={(e) => setSceneId(e.target.value)}
          />
          <input
            type="text"
            placeholder="技能名"
            value={skillName}
            disabled={recognitionType === "presence"}
            onChange={(e) => setSkillName(e.target.value)}
          />
          <select
            value={recognitionType}
            onChange={(e) => setRecognitionType(e.target.value)}
          >
            <option value="">全部类型</option>
            <option value="01">01 人员计数（入）</option>
            <option value="02">02 人员计数（出）</option>
            <option value="presence">画面人数</option>
          </select>
          <input
            type="datetime-local"
            title="起始时间"
            value={timeFrom}
            onChange={(e) => setTimeFrom(e.target.value)}
          />
          <span className="filter-sep">至</span>
          <input
            type="datetime-local"
            title="结束时间"
            value={timeTo}
            onChange={(e) => setTimeTo(e.target.value)}
          />
          <select value={status} onChange={(e) => setStatus(e.target.value)}>
            <option value="">全部状态</option>
            <option value="new">new</option>
            <option value="acked">acked</option>
            <option value="closed">closed</option>
          </select>
          <button
            className="btn"
            type="button"
            onClick={() => {
              setSceneId("");
              setSkillName("");
              setRecognitionType("");
              setTimeFrom("");
              setTimeTo("");
              setStatus("");
            }}
          >
            清空
          </button>
          <button className="btn" onClick={() => load(page)}>
            刷新
          </button>
        </div>
      </div>

      {error && <p className="error">{error}</p>}

      <div className="card page-list-card">
        <div className="page-list-scroll">
          <table>
            <thead>
              <tr>
                <th>时间</th>
                <th>场景</th>
                <th>类型</th>
                <th>人数</th>
                <th>截图</th>
                <th>状态</th>
                <th>操作</th>
              </tr>
            </thead>
            <tbody>
              {items.map((a) => {
                const hasImage = !!(a.image_url || a.payload?.pic_url);
                return (
                  <tr key={a.id}>
                    <td className="mono">{a.created_at}</td>
                    <td>
                      {a.scene_id}
                      <div className="muted mono">{a.skill_name || "-"}</div>
                    </td>
                    <td className="mono">{typeLabel(a)}</td>
                    <td>
                      count={a.count} / enter={a.enter_count}
                    </td>
                    <td>
                      <button
                        className="btn icon"
                        type="button"
                        title={hasImage ? "查看事件截图" : "暂无截图"}
                        disabled={!hasImage}
                        onClick={() => setPreview(a)}
                      >
                        <EvidenceIcon />
                      </button>
                    </td>
                    <td>
                      <span className={`badge ${a.status === "new" ? "run" : "stop"}`}>
                        {a.status}
                      </span>
                    </td>
                    <td className="row-actions">
                      <button className="btn" onClick={() => setSelected(a)}>
                        详情
                      </button>
                      {a.status === "new" && (
                        <button
                          className="btn primary"
                          onClick={async () => {
                            await api.updateEvent(a.id, "acked");
                            await load(page);
                          }}
                        >
                          确认
                        </button>
                      )}
                      <button
                        className="btn"
                        onClick={async () => {
                          await api.updateEvent(a.id, "closed");
                          await load(page);
                        }}
                      >
                        关闭
                      </button>
                    </td>
                  </tr>
                );
              })}
              {!items.length && (
                <tr>
                  <td colSpan={7} className="muted">
                    暂无事件。正常过线或画面人数变化后会自动写入。
                  </td>
                </tr>
              )}
            </tbody>
          </table>

          {selected && (
            <div className="card" style={{ marginTop: 12 }}>
              <div className="page-head" style={{ marginBottom: 8 }}>
                <h1 style={{ fontSize: 18 }}>事件详情 #{selected.id}</h1>
                <button className="btn ghost" onClick={() => setSelected(null)}>
                  关闭
                </button>
              </div>
              <p className="mono">{selected.message}</p>
              {selected.image_url && (
                <p>
                  <a href={selected.image_url} target="_blank" rel="noreferrer">
                    打开事件图片
                  </a>
                </p>
              )}
              <pre
                className="mono"
                style={{
                  whiteSpace: "pre-wrap",
                  background: "#101613",
                  padding: 12,
                  borderRadius: 8,
                  maxHeight: 360,
                  overflow: "auto",
                }}
              >
                {JSON.stringify(selected.payload || selected, null, 2)}
              </pre>
            </div>
          )}
        </div>
        <Pager
          page={page}
          pageSize={PAGE_SIZE}
          total={total}
          onChange={setPage}
        />
      </div>

      {preview && (
        <div className="modal-backdrop" onClick={() => setPreview(null)} role="presentation">
          <div
            className="modal-panel evidence-panel"
            onClick={(e) => e.stopPropagation()}
            role="dialog"
            aria-modal="true"
            aria-label="事件截图"
          >
            <div className="modal-head">
              <div>
                <h2>事件截图 #{preview.id}</h2>
                <p className="muted mono">
                  {preview.created_at} · {preview.scene_id} / {preview.skill_name || "-"}
                </p>
              </div>
              <button className="btn" type="button" onClick={() => setPreview(null)}>
                关闭
              </button>
            </div>
            {(preview.image_url || preview.payload?.pic_url) ? (
              <img
                className="evidence-image"
                src={preview.image_url || preview.payload?.pic_url}
                alt="事件截图"
              />
            ) : (
              <p className="muted">暂无截图</p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
