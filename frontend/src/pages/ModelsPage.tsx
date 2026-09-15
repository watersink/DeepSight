import { useEffect, useMemo, useState } from "react";
import { api } from "../api";
import Pager from "../components/Pager";

const PAGE_SIZE = 10;
const STATUS_REFRESH_DELAY_MS = 600;
const STATUS_REFRESH_RETRY_MS = 1200;

function workerLabel(m: any): string {
  const workers = m?.workers || [];
  if (!workers.length) return "相关 Worker";
  return workers
    .map((w: any) => w.worker_name || w.worker_id)
    .filter(Boolean)
    .join("、");
}

/** 模型运行状态：就绪 / 离线 */
function modelOnline(m: any): boolean {
  return Boolean(m?.ready);
}

function statusBadge(online: boolean) {
  return (
    <span className={`badge ${online ? "run" : "stop"}`}>
      {online ? "就绪" : "离线"}
    </span>
  );
}

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export default function ModelsPage() {
  const [data, setData] = useState<any | null>(null);
  const [error, setError] = useState("");
  const [msg, setMsg] = useState("");
  const [loading, setLoading] = useState(false);
  const [acting, setActing] = useState("");
  const [page, setPage] = useState(1);

  const fetchModels = async () => {
    const res = await api.getModels();
    setData(res);
    if (res.error) setError(res.error);
    else setError("");
    return res;
  };

  const load = async (keepPage = false) => {
    setLoading(true);
    setError("");
    try {
      await fetchModels();
      if (!keepPage) setPage(1);
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

  /** 操作后刷新列表，直到目标模型状态符合预期或重试耗尽 */
  const refreshUntilStatus = async (
    modelName: string,
    expectOnline: boolean
  ) => {
    await sleep(STATUS_REFRESH_DELAY_MS);
    let res = await fetchModels();
    const hit = (res.items || []).find(
      (m: any) => String(m?.name || "") === modelName
    );
    if (hit && modelOnline(hit) === expectOnline) return;
    await sleep(STATUS_REFRESH_RETRY_MS);
    await fetchModels();
  };

  const runAction = async (
    action: "load" | "unload" | "delete",
    model: any
  ) => {
    const name = String(model?.name || "").trim();
    if (!name) return;
    const nodes = workerLabel(model);
    const tips: Record<string, string> = {
      load: `确认在 ${nodes} 上加载模型「${name}」？\n（调用 Triton Repository Load API）`,
      unload: `确认在 ${nodes} 上卸载模型「${name}」？\n（调用 Triton Repository Unload API，仓库文件保留，可再加载）`,
      delete:
        `确认删除模型「${name}」？\n` +
        `Triton 无删文件接口；将调用 Unload API 并卸载依赖（unload_dependents）。\n` +
        `目标节点：${nodes}`,
    };
    if (!confirm(tips[action])) return;

    const key = `${action}:${name}`;
    setActing(key);
    setError("");
    setMsg("");
    const expectOnline = action === "load";
    try {
      let res: any;
      if (action === "load") {
        res = await api.loadModel(name);
        setMsg(`已加载：${name}`);
      } else if (action === "unload") {
        res = await api.unloadModel(name);
        setMsg(`已卸载：${name}`);
      } else {
        res = await api.deleteModel(name);
        setMsg(`已删除（卸载及依赖）：${name}`);
      }
      if (res?.ok === false || (res?.errors && res.errors.length)) {
        throw new Error(
          Array.isArray(res.errors) ? res.errors.join("; ") : "操作未完全成功"
        );
      }
      // 先乐观更新当前行状态，再向服务端拉取确认
      setData((prev: any) => {
        if (!prev?.items) return prev;
        return {
          ...prev,
          items: prev.items.map((m: any) =>
            String(m?.name || "") === name
              ? {
                  ...m,
                  ready: expectOnline,
                  state: expectOnline ? "READY" : "UNAVAILABLE",
                  workers: (m.workers || []).map((w: any) => ({
                    ...w,
                    ready: expectOnline,
                    state: expectOnline ? "READY" : "UNAVAILABLE",
                  })),
                }
              : m
          ),
        };
      });
      await refreshUntilStatus(name, expectOnline);
    } catch (e: any) {
      setError(e.message || String(e));
      try {
        await fetchModels();
      } catch {
        /* ignore */
      }
    } finally {
      setActing("");
    }
  };

  return (
    <div className="page-shell">
      <div className="page-head">
        <div>
          <h1>模型管理</h1>
          <p>
            汇总各算力 Worker 上 Triton 已部署的模型，并标注部署在哪台机器。
            加载 / 卸载基于 Triton 自带 Repository API；删除对应卸载并包含依赖。
          </p>
        </div>
        <button className="btn" onClick={() => load()} disabled={loading || !!acting}>
          {loading ? "刷新中…" : "刷新"}
        </button>
      </div>

      {error && <p className="error">{error}</p>}
      {msg && <p className="ok">{msg}</p>}

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
                <th>状态</th>
                <th>部署机器</th>
                <th>操作</th>
              </tr>
            </thead>
            <tbody>
              {paged.map((m: any) => {
                const name = String(m.name || "");
                const online = modelOnline(m);
                const rowBusy =
                  acting === `load:${name}` ||
                  acting === `unload:${name}` ||
                  acting === `delete:${name}`;
                return (
                  <tr key={`${m.name}:${m.version || ""}`}>
                    <td className="mono">{m.name}</td>
                    <td className="mono">{m.version || "latest"}</td>
                    <td className="mono muted">{m.state || "-"}</td>
                    <td>{statusBadge(online)}</td>
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
                              {w.ready ? " · 就绪" : " · 离线"}
                            </span>
                          ))}
                        </div>
                      ) : (
                        <span className="muted">—</span>
                      )}
                    </td>
                    <td className="row-actions">
                      <button
                        className="btn"
                        type="button"
                        disabled={!!acting || loading}
                        onClick={() => runAction("load", m)}
                        title="Triton Repository Load"
                      >
                        {rowBusy && acting.startsWith("load:")
                          ? "加载中…"
                          : "加载"}
                      </button>
                      <button
                        className="btn"
                        type="button"
                        disabled={!!acting || loading}
                        onClick={() => runAction("unload", m)}
                        title="Triton Repository Unload"
                      >
                        {rowBusy && acting.startsWith("unload:")
                          ? "卸载中…"
                          : "卸载"}
                      </button>
                      <button
                        className="btn danger"
                        type="button"
                        disabled={!!acting || loading}
                        onClick={() => runAction("delete", m)}
                        title="Unload + unload_dependents"
                      >
                        {rowBusy && acting.startsWith("delete:")
                          ? "删除中…"
                          : "删除"}
                      </button>
                    </td>
                  </tr>
                );
              })}
              {!paged.length && (
                <tr>
                  <td colSpan={6} className="muted">
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
