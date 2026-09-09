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
  const paged = useMemo(() => {
    const start = (page - 1) * PAGE_SIZE;
    return items.slice(start, start + PAGE_SIZE);
  }, [items, page]);

  return (
    <div className="page-shell">
      <div className="page-head">
        <div>
          <h1>模型管理</h1>
          <p>从 Triton Inference Server 拉取本机已部署模型（repository index）。</p>
        </div>
        <button className="btn" onClick={load} disabled={loading}>
          {loading ? "刷新中…" : "刷新"}
        </button>
      </div>

      {error && <p className="error">{error}</p>}

      <div className="card page-meta-card">
        <div className="form-grid">
          <label>
            Triton 地址
            <input readOnly value={data?.server_url || "-"} />
          </label>
          <label>
            服务状态
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
            服务名
            <input readOnly value={data?.server_name || "-"} />
          </label>
          <label>
            Triton 版本
            <input readOnly value={data?.server_version || "-"} />
          </label>
        </div>
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
                </tr>
              ))}
              {!paged.length && (
                <tr>
                  <td colSpan={4} className="muted">
                    {loading
                      ? "加载中…"
                      : "暂无模型。请确认 Triton 已启动且仓库中有模型。"}
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
