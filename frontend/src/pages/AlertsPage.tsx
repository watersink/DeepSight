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

function resolveVideoUrl(a: any): string | null {
  return (
    a?.video_url ||
    a?.payload?.video_minio_url ||
    a?.payload?.video_url ||
    null
  );
}

function EvidenceModal({
  alert,
  onClose,
}: {
  alert: any;
  onClose: () => void;
}) {
  const imageUrl = alert.image_url || alert.payload?.image_minio_url || alert.payload?.pic_url;
  const videoUrl = resolveVideoUrl(alert);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div className="modal-backdrop" onClick={onClose} role="presentation">
      <div
        className="modal-panel evidence-panel"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-label="报警证据"
      >
        <div className="modal-head">
          <div>
            <h2>报警证据 #{alert.id}</h2>
            <p className="muted mono">
              {alert.created_at} · {alert.scene_id} / {alert.skill_name || "-"}
            </p>
          </div>
          <button className="btn" type="button" onClick={onClose}>
            关闭
          </button>
        </div>
        <div className="evidence-grid">
          <section>
            <h3 className="section-title">报警图片</h3>
            {imageUrl ? (
              <a href={imageUrl} target="_blank" rel="noreferrer">
                <img className="evidence-image" src={imageUrl} alt="报警图片" />
              </a>
            ) : (
              <p className="muted">暂无报警图片</p>
            )}
          </section>
          <section>
            <h3 className="section-title">报警视频（约 10 秒）</h3>
            {videoUrl ? (
              <video
                className="evidence-video"
                src={videoUrl}
                controls
                autoPlay
                muted
                playsInline
              />
            ) : (
              <p className="muted">
                暂无报警视频。新告警会在触发后异步截取约 10 秒证据视频，请稍后刷新。
              </p>
            )}
          </section>
        </div>
        {alert.message && (
          <p className="muted" style={{ marginTop: 12 }}>
            {alert.message}
          </p>
        )}
      </div>
    </div>
  );
}

export default function AlertsPage() {
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
  const [evidence, setEvidence] = useState<any | null>(null);

  const filterKey = `${status}|${sceneId}|${skillName}|${recognitionType}|${timeFrom}|${timeTo}`;

  const load = async (pageNo = page) => {
    setError("");
    try {
      const data = await api.getAlerts({
        status: status || undefined,
        sceneId: sceneId.trim() || undefined,
        skillName: skillName.trim() || undefined,
        recognitionType: recognitionType || undefined,
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

  return (
    <div className="page-shell">
      <div className="page-head">
        <div>
          <h1>报警管理</h1>
          <p>仅展示违规报警（04–07：绕行/闸机反向）；正常过线与画面人数见「事件管理」。</p>
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
            onChange={(e) => setSkillName(e.target.value)}
          />
          <select
            value={recognitionType}
            onChange={(e) => setRecognitionType(e.target.value)}
          >
            <option value="">全部类型</option>
            <option value="04">04 非常规通道入井</option>
            <option value="05">05 非常规通道出井</option>
            <option value="06">06 入井闸机出闸</option>
            <option value="07">07 出井闸机入闸</option>
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
                <th>证据</th>
                <th>状态</th>
                <th>操作</th>
              </tr>
            </thead>
            <tbody>
              {items.map((a) => {
                const hasEvidence = !!(
                  a.image_url ||
                  resolveVideoUrl(a) ||
                  a.payload?.pic_url
                );
                return (
                  <tr key={a.id}>
                    <td className="mono">{a.created_at}</td>
                    <td>
                      {a.scene_id}
                      <div className="muted mono">{a.skill_name || "-"}</div>
                    </td>
                    <td className="mono">
                      {(a.recognition_types || []).join(",") || "-"}
                    </td>
                    <td>
                      count={a.count} / enter={a.enter_count}
                    </td>
                    <td>
                      <button
                        className="btn icon"
                        type="button"
                        title={hasEvidence ? "查看报警证据" : "暂无证据素材"}
                        disabled={!hasEvidence}
                        onClick={() => setEvidence(a)}
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
                            await api.updateAlert(a.id, "acked");
                            await load(page);
                          }}
                        >
                          确认
                        </button>
                      )}
                      <button
                        className="btn"
                        onClick={async () => {
                          await api.updateAlert(a.id, "closed");
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
                    暂无报警。产生告警后会自动写入 MySQL。
                  </td>
                </tr>
              )}
            </tbody>
          </table>

          {selected && (
            <div className="card" style={{ marginTop: 12 }}>
              <div className="page-head" style={{ marginBottom: 8 }}>
                <h1 style={{ fontSize: 18 }}>报警详情 #{selected.id}</h1>
                <button className="btn ghost" onClick={() => setSelected(null)}>
                  关闭
                </button>
              </div>
              <p className="mono">{selected.message}</p>
              {selected.image_url && (
                <p>
                  <a href={selected.image_url} target="_blank" rel="noreferrer">
                    打开告警图片
                  </a>
                </p>
              )}
              {resolveVideoUrl(selected) && (
                <p>
                  <a href={resolveVideoUrl(selected)!} target="_blank" rel="noreferrer">
                    打开告警视频
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

      {evidence && (
        <EvidenceModal alert={evidence} onClose={() => setEvidence(null)} />
      )}
    </div>
  );
}
