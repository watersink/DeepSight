import { useEffect, useMemo, useState } from "react";
import { api } from "../api";
import Pager from "../components/Pager";

const PAGE_SIZE = 10;

export default function ModelsPage() {
  const [data, setData] = useState<any | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [page, setPage] = useState(1);

  const load = async () => {
    setLoading(true);
    setError("");
    try {
      const res = await api.getModels();
      setData(res);
      if (res.error) setError(res.error);
      setPage(1);
    } catch (e: any) {
      setError(e.message || String(e));
      setData(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const items = data?.items || [];
  const workers = data?.workers || [];
  const paged = useMemo(() => {
    const start = (page - 1) * PAGE_SIZE;
    return items.slice(start, start + PAGE_SIZE);
  }, [items, page]);

  return (
    <div className="page-shell">
      <div className="page-head">
        <div>
          <h1>模型管理</h1>
          <p>
            汇总各算力 Worker 上 Triton 已部署的模型，并标注部署在哪台机器。
          </p>
        </div>
        <button className="btn" onClick={load} disabled={loading}>
          {loading ? "刷新中…" : "刷新"}
        </button>
      </div>

      {error && <p className="error">{error}</p>}

      <div className="card page-meta-card">
        <div className="form-grid">
          <label>
            Triton 地址（汇总）
            <input readOnly value={data?.server_url || "-"} />
          </label>
          <label>
            整体状态
            <input
              readOnly
              value={
                data
                  ? `live=${data.server_live ? "是" : "否"} / ready=${
                      data.server_ready ? "是" : "否"
                    }`
                  : "-"
              }
            />
          </label>
          <label>
            Worker 节点数
            <input readOnly value={String(workers.length || 0)} />
          </label>
          <label>
            模型数
            <input readOnly value={String(items.length || 0)} />
          </label>
        </div>
        {!!workers.length && (
          <div className="worker-tag-row" style={{ marginTop: 12 }}>
            {workers.map((w: any) => (
              <span
                key={w.worker_id}
                className={`badge worker-tag ${
                  w.online === false ? "stop" : w.online ? "run" : ""
                }`}
                title={w.server_url || w.error || w.worker_id}
              >
                {w.worker_name || w.worker_id}
                {w.online === false ? " · 离线" : ""}
                {w.server_url ? ` · ${w.server_url}` : ""}
              </span>
            ))}
          </div>
        )}
      </div>

      <div className="card page-list-card">
        <div className="page-list-scroll">
          <table>
            <thead>
              <tr>
                <th>模型名</th>
                <th>版本</th>
                <th>仓库状态</th>
                <th>就绪</th>
                <th>部署机器</th>
              </tr>
            </thead>
            <tbody>
              {paged.map((m: any) => (
                <tr key={`${m.name}:${m.version || ""}`}>
                  <td className="mono">{m.name}</td>
                  <td className="mono">{m.version || "latest"}</td>
                  <td className="mono muted">{m.state || "-"}</td>
                  <td>
                    <span className={`badge ${m.ready ? "run" : "stop"}`}>
                      {m.ready ? "就绪" : "未就绪"}
                    </span>
                  </td>
                  <td>
                    {(m.workers || []).length ? (
                      <div className="worker-tag-row">
                        {(m.workers || []).map((w: any) => (
                          <span
                            key={w.worker_id}
                            className={`badge worker-tag ${
                              w.ready ? "run" : "stop"
                            }`}
                            title={w.triton_url || w.worker_id}
                          >
                            {w.worker_name || w.worker_id}
                            {w.triton_url ? ` · ${w.triton_url}` : ""}
                            {w.ready ? "" : " · 未就绪"}
                          </span>
                        ))}
                      </div>
                    ) : (
                      <span className="muted">—</span>
                    )}
                  </td>
                </tr>
              ))}
              {!paged.length && (
                <tr>
                  <td colSpan={5} className="muted">
                    {loading
                      ? "加载中…"
                      : "暂无模型。请确认各 Worker 的 Triton 已启动且仓库中有模型。"}
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
        <Pager
          page={page}
          pageSize={PAGE_SIZE}
          total={items.length}
          onChange={setPage}
        />
      </div>
    </div>
  );
}
