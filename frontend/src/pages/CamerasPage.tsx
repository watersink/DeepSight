import { FormEvent, MouseEvent, ReactNode, useEffect, useMemo, useState } from "react";
import { api } from "../api";
import Pager from "../components/Pager";
import StreamPlayerModal from "../components/StreamPlayerModal";

type Sel =
  | { type: "mine"; mineId: number }
  | { type: "site"; mineId: number; siteId: number }
  | { type: "unassigned" }
  | null;

type CtxMenu =
  | {
      kind: "mine";
      x: number;
      y: number;
      mine: any;
    }
  | {
      kind: "site";
      x: number;
      y: number;
      mineId: number;
      site: any;
    }
  | {
      kind: "camera";
      x: number;
      y: number;
      camera: any;
    }
  | {
      kind: "blank";
      x: number;
      y: number;
    }
  | null;

type Dialog =
  | { type: "mine"; mode: "create" | "edit"; mine?: any }
  | { type: "site"; mode: "create" | "edit"; mineId: number; site?: any }
  | { type: "camera"; mode: "create" | "edit"; siteId?: number | null; camera?: any }
  | null;

const ONLINE_POLL_MS = 60_000;
const PAGE_SIZE = 10;

function OnlineBadge({ cam }: { cam: any }) {
  if (cam.online === true) return <span className="badge run">在线</span>;
  if (cam.online === false) return <span className="badge stop">离线</span>;
  return (
    <span className="badge" title={cam.online_check_error || "未知"}>
      未知
    </span>
  );
}

function FormDialog({
  title,
  onClose,
  onSubmit,
  children,
  submitText,
  width,
}: {
  title: string;
  onClose: () => void;
  onSubmit: (e: FormEvent) => void;
  children: ReactNode;
  submitText: string;
  width?: string;
}) {
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
        className="modal-panel"
        style={{ width: width || "min(560px, 100%)" }}
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
      >
        <div className="modal-head">
          <h2>{title}</h2>
          <button className="btn" type="button" onClick={onClose}>
            关闭
          </button>
        </div>
        <form onSubmit={onSubmit}>
          {children}
          <div className="toolbar" style={{ marginTop: 12 }}>
            <button className="btn primary" type="submit">
              {submitText}
            </button>
            <button className="btn ghost" type="button" onClick={onClose}>
              取消
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

function formatDataTime(d = new Date()): string {
  const p = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(
    d.getHours()
  )}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
}

function analysisTypeLabel(code: string): string {
  if (code === "01") return "01 人员计数（入）";
  if (code === "02") return "02 人员计数（出）";
  return code || "—";
}

const emptyCamDraft = () => ({
  name: "",
  ingest_mode: "push" as "push" | "proxy",
  in_url: "",
  source_url: "",
  zlm_app: "",
  zlm_stream: "",
  camera_code: "",
  mine_code: "",
  position_type: "",
  position_desc: "",
  ps_station_code: "",
  analysis_type: "01" as "01" | "02",
  data_time: formatDataTime(),
  basic_image_base64: "",
  has_basic_image: false,
  site_id: null as number | null,
  enabled: true,
  remark: "",
});

export default function CamerasPage() {
  const [tree, setTree] = useState<{ mines: any[]; unassigned: any[] }>({
    mines: [],
    unassigned: [],
  });
  const [sel, setSel] = useState<Sel>(null);
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [lastCheck, setLastCheck] = useState("");
  const [playing, setPlaying] = useState<any | null>(null);
  const [treeCollapsed, setTreeCollapsed] = useState(false);
  const [ctx, setCtx] = useState<CtxMenu>(null);
  const [dialog, setDialog] = useState<Dialog>(null);
  const [page, setPage] = useState(1);

  const [mineDraft, setMineDraft] = useState({
    name: "",
    code: "",
    remark: "",
    enabled: true,
  });
  const [siteDraft, setSiteDraft] = useState({
    name: "",
    code: "",
    remark: "",
    enabled: true,
  });
  const [camDraft, setCamDraft] = useState(emptyCamDraft);
  const load = async () => {
    setLoading(true);
    setError("");
    try {
      const data = await api.getCameraTree();
      setTree({ mines: data.mines || [], unassigned: data.unassigned || [] });
      setLastCheck(new Date().toLocaleTimeString());
      setExpanded((prev) => {
        const next = { ...prev };
        for (const m of data.mines || []) {
          if (next[`m-${m.id}`] === undefined) next[`m-${m.id}`] = true;
          for (const s of m.sites || []) {
            if (next[`s-${s.id}`] === undefined) next[`s-${s.id}`] = true;
          }
        }
        return next;
      });
    } catch (e: any) {
      setError(e.message || String(e));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
    const timer = setInterval(load, ONLINE_POLL_MS);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    if (!ctx) return;
    const close = () => setCtx(null);
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") close();
    };
    window.addEventListener("click", close);
    window.addEventListener("scroll", close, true);
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("click", close);
      window.removeEventListener("scroll", close, true);
      window.removeEventListener("keydown", onKey);
    };
  }, [ctx]);

  const selectedCameras = useMemo(() => {
    if (!sel) return [];
    if (sel.type === "unassigned") return tree.unassigned;
    if (sel.type === "mine") {
      const mine = tree.mines.find((m) => m.id === sel.mineId);
      return (mine?.sites || []).flatMap((s: any) => s.cameras || []);
    }
    const mine = tree.mines.find((m) => m.id === sel.mineId);
    const site = (mine?.sites || []).find((s: any) => s.id === sel.siteId);
    return site?.cameras || [];
  }, [sel, tree]);

  useEffect(() => {
    setPage(1);
  }, [sel]);

  const totalPages = Math.max(1, Math.ceil(selectedCameras.length / PAGE_SIZE));
  const safePage = Math.min(page, totalPages);
  const pagedCameras = useMemo(() => {
    const start = (safePage - 1) * PAGE_SIZE;
    return selectedCameras.slice(start, start + PAGE_SIZE);
  }, [selectedCameras, safePage]);

  const selectedTitle = useMemo(() => {
    if (!sel) return "请选择左侧煤矿或地点";
    if (sel.type === "unassigned") return "未归类摄像头";
    const mine = tree.mines.find((m) => m.id === sel.mineId);
    if (sel.type === "mine") return `煤矿 · ${mine?.name || sel.mineId}`;
    const site = (mine?.sites || []).find((s: any) => s.id === sel.siteId);
    return `${mine?.name || ""} / ${site?.name || sel.siteId}`;
  }, [sel, tree]);

  const siteOptions = useMemo(() => {
    const opts: { id: number; label: string }[] = [];
    for (const m of tree.mines) {
      for (const s of m.sites || []) {
        opts.push({ id: s.id, label: `${m.name} / ${s.name}` });
      }
    }
    return opts;
  }, [tree]);

  const openMineDialog = (mode: "create" | "edit", mine?: any) => {
    setMineDraft({
      name: mine?.name || "",
      code: mine?.code || "",
      remark: mine?.remark || "",
      enabled: mine ? !!mine.enabled : true,
    });
    setDialog({ type: "mine", mode, mine });
    setCtx(null);
  };

  const openSiteDialog = (
    mode: "create" | "edit",
    mineId: number,
    site?: any
  ) => {
    setSiteDraft({
      name: site?.name || "",
      code: site?.code || "",
      remark: site?.remark || "",
      enabled: site ? !!site.enabled : true,
    });
    setDialog({ type: "site", mode, mineId, site });
    setCtx(null);
  };

  const openCameraDialog = async (
    mode: "create" | "edit",
    opts?: { siteId?: number | null; camera?: any }
  ) => {
    const camera = opts?.camera;
    let detail = camera;
    if (mode === "edit" && camera?.id) {
      try {
        detail = await api.getCamera(camera.id);
      } catch {
        detail = camera;
      }
    }
    setCamDraft({
      name: detail?.name || "",
      ingest_mode: detail?.ingest_mode === "proxy" ? "proxy" : "push",
      in_url: detail?.in_url || "",
      source_url: detail?.source_url || "",
      zlm_app: detail?.zlm_app || "",
      zlm_stream: detail?.zlm_stream || "",
      camera_code: detail?.camera_code || "",
      mine_code: detail?.mine_code || "",
      position_type: detail?.position_type || "",
      position_desc: detail?.position_desc || "",
      ps_station_code: detail?.ps_station_code || "",
      analysis_type:
        detail?.analysis_type === "02" ? "02" : "01",
      data_time: detail?.data_time || formatDataTime(),
      basic_image_base64: detail?.basic_image_base64 || "",
      has_basic_image: !!(
        detail?.has_basic_image || detail?.basic_image_base64
      ),
      site_id:
        detail?.site_id ??
        opts?.siteId ??
        (sel?.type === "site" ? sel.siteId : null),
      enabled: detail ? !!detail.enabled : true,
      remark: detail?.remark || "",
    });
    setDialog({
      type: "camera",
      mode,
      siteId: opts?.siteId ?? detail?.site_id ?? null,
      camera: detail || camera,
    });
    setCtx(null);
  };

  const saveMine = async (e: FormEvent) => {
    e.preventDefault();
    if (!dialog || dialog.type !== "mine") return;
    setError("");
    try {
      if (dialog.mode === "edit" && dialog.mine) {
        await api.updateMine(dialog.mine.id, mineDraft);
      } else {
        await api.createMine(mineDraft);
      }
      setDialog(null);
      await load();
    } catch (err: any) {
      setError(err.message || String(err));
    }
  };

  const saveSite = async (e: FormEvent) => {
    e.preventDefault();
    if (!dialog || dialog.type !== "site") return;
    setError("");
    try {
      if (dialog.mode === "edit" && dialog.site) {
        await api.updateSite(dialog.site.id, {
          ...siteDraft,
          mine_id: dialog.mineId,
        });
      } else {
        await api.createSite({
          mine_id: dialog.mineId,
          ...siteDraft,
        });
      }
      setDialog(null);
      await load();
    } catch (err: any) {
      setError(err.message || String(err));
    }
  };

  const saveCamera = async (e: FormEvent) => {
    e.preventDefault();
    if (!dialog || dialog.type !== "camera") return;
    setError("");
    try {
      if (!camDraft.camera_code.trim()) {
        setError("请填写摄像仪编码");
        return;
      }
      if (!camDraft.position_type.trim()) {
        setError("请填写安装位置分类编码");
        return;
      }
      if (!camDraft.position_desc.trim()) {
        setError("请填写安装位置描述");
        return;
      }
      const body: Record<string, unknown> = {
        name: camDraft.name,
        ingest_mode: camDraft.ingest_mode,
        in_url: camDraft.in_url,
        source_url:
          camDraft.ingest_mode === "proxy"
            ? camDraft.source_url || null
            : null,
        zlm_app: camDraft.zlm_app,
        zlm_stream: camDraft.zlm_stream,
        camera_code: camDraft.camera_code.trim(),
        mine_code: camDraft.mine_code.trim(),
        position_type: camDraft.position_type.trim(),
        position_desc: camDraft.position_desc.trim(),
        ps_station_code: camDraft.ps_station_code.trim(),
        analysis_type: camDraft.analysis_type,
        data_time: camDraft.data_time.trim() || formatDataTime(),
        site_id: camDraft.site_id,
        enabled: camDraft.enabled,
        remark: camDraft.remark,
      };
      // 仅在用户显式填写/保留了模板时提交；否则由服务端保存后自动 ZLM 截图
      if (camDraft.basic_image_base64.trim()) {
        body.basic_image_base64 = camDraft.basic_image_base64.trim();
      } else if (
        dialog.mode === "edit" &&
        dialog.camera?.has_basic_image &&
        !camDraft.has_basic_image
      ) {
        // 用户清除了已有模板 → 通知服务端清空并自动重截
        body.basic_image_base64 = "";
      }
      if (dialog.mode === "edit" && dialog.camera) {
        await api.updateCamera(dialog.camera.id, body);
      } else {
        if (!body.site_id) {
          setError("请选择归属地点");
          return;
        }
        await api.createCamera(body);
      }
      setDialog(null);
      await load();
    } catch (err: any) {
      setError(err.message || String(err));
    }
  };

  const deleteMine = async (mine: any) => {
    if (!confirm(`删除煤矿「${mine.name}」及其地点？摄像头会变为未归类。`)) return;
    setError("");
    try {
      await api.deleteMine(mine.id);
      if (sel && "mineId" in sel && sel.mineId === mine.id) setSel(null);
      setCtx(null);
      await load();
    } catch (err: any) {
      setError(err.message || String(err));
    }
  };

  const deleteSite = async (site: any, mineId: number) => {
    if (!confirm(`删除地点「${site.name}」？其下摄像头将变为未归类。`)) return;
    setError("");
    try {
      await api.deleteSite(site.id);
      if (sel?.type === "site" && sel.siteId === site.id) {
        setSel({ type: "mine", mineId });
      }
      setCtx(null);
      await load();
    } catch (err: any) {
      setError(err.message || String(err));
    }
  };

  const deleteCamera = async (camera: any) => {
    if (!confirm(`删除摄像头「${camera.name}」？`)) return;
    setError("");
    try {
      await api.deleteCamera(camera.id);
      setCtx(null);
      await load();
    } catch (err: any) {
      setError(err.message || String(err));
    }
  };

  const showCtx = (
    e: MouseEvent,
    menu: Exclude<CtxMenu, null>
  ) => {
    e.preventDefault();
    e.stopPropagation();
    const pad = 8;
    const approxW = 168;
    const approxH = 140;
    const x = Math.min(e.clientX, window.innerWidth - approxW - pad);
    const y = Math.min(e.clientY, window.innerHeight - approxH - pad);
    setCtx({ ...menu, x: Math.max(pad, x), y: Math.max(pad, y) });
  };

  return (
    <div className="page-shell">
      <div className="page-head">
        <div>
          <h1>摄像头配置</h1>
          <p>
            左键选中查看列表；右键煤矿 / 地点 / 摄像头进行新增、编辑、删除。
            {lastCheck ? ` 上次刷新 ${lastCheck}` : ""}
          </p>
        </div>
        <div className="toolbar">
          <button className="btn primary" type="button" onClick={() => openMineDialog("create")}>
            新增煤矿
          </button>
          <button className="btn" onClick={load} disabled={loading}>
            {loading ? "检测中…" : "刷新"}
          </button>
        </div>
      </div>

      {error && <p className="error">{error}</p>}

      <div className={`camera-layout ${treeCollapsed ? "tree-collapsed" : ""}`}>
        <aside
          className="camera-tree card"
          onContextMenu={(e) => {
            // 空白区域右键：新增煤矿
            if ((e.target as HTMLElement).closest(".tree-node, .tree-toggle")) return;
            showCtx(e, { kind: "blank", x: e.clientX, y: e.clientY });
          }}
        >
          <div className="tree-panel-head">
            <div className="tree-section-title">组织树（右键操作）</div>
            <button
              type="button"
              className="tree-collapse-btn"
              title="收起组织树"
              aria-label="收起组织树"
              onClick={() => setTreeCollapsed(true)}
            >
              «
            </button>
          </div>

          <button
            type="button"
            className={`tree-node ${sel?.type === "unassigned" ? "active" : ""}`}
            onClick={() => setSel({ type: "unassigned" })}
          >
            未归类
            <span className="muted">({tree.unassigned.length})</span>
          </button>

          {tree.mines.map((m) => {
            const mKey = `m-${m.id}`;
            const open = !!expanded[mKey];
            const camCount = (m.sites || []).reduce(
              (n: number, s: any) => n + (s.cameras?.length || 0),
              0
            );
            return (
              <div key={m.id} className="tree-mine">
                <div className="tree-row">
                  <button
                    type="button"
                    className="tree-toggle"
                    onClick={() =>
                      setExpanded((p) => ({ ...p, [mKey]: !p[mKey] }))
                    }
                  >
                    {open ? "▾" : "▸"}
                  </button>
                  <button
                    type="button"
                    className={`tree-node ${
                      sel?.type === "mine" && sel.mineId === m.id ? "active" : ""
                    }`}
                    onClick={() => setSel({ type: "mine", mineId: m.id })}
                    onContextMenu={(e) =>
                      showCtx(e, { kind: "mine", x: 0, y: 0, mine: m })
                    }
                  >
                    {m.name}
                    <span className="muted mono"> {m.code || ""}</span>
                    <span className="muted"> · {camCount}路</span>
                  </button>
                </div>

                {open &&
                  (m.sites || []).map((s: any) => {
                    const sKey = `s-${s.id}`;
                    const sOpen = !!expanded[sKey];
                    return (
                      <div key={s.id}>
                        <div className="tree-row tree-site">
                          <button
                            type="button"
                            className="tree-toggle"
                            onClick={() =>
                              setExpanded((p) => ({ ...p, [sKey]: !p[sKey] }))
                            }
                          >
                            {sOpen ? "▾" : "▸"}
                          </button>
                          <button
                            type="button"
                            className={`tree-node ${
                              sel?.type === "site" && sel.siteId === s.id
                                ? "active"
                                : ""
                            }`}
                            onClick={() =>
                              setSel({
                                type: "site",
                                mineId: m.id,
                                siteId: s.id,
                              })
                            }
                            onContextMenu={(e) =>
                              showCtx(e, {
                                kind: "site",
                                x: 0,
                                y: 0,
                                mineId: m.id,
                                site: s,
                              })
                            }
                          >
                            {s.name}
                            <span className="muted">
                              {" "}
                              ({(s.cameras || []).length})
                            </span>
                          </button>
                        </div>
                        {sOpen &&
                          (s.cameras || []).map((c: any) => (
                            <button
                              key={c.id}
                              type="button"
                              className="tree-node tree-camera"
                              onClick={() =>
                                setSel({
                                  type: "site",
                                  mineId: m.id,
                                  siteId: s.id,
                                })
                              }
                              onContextMenu={(e) =>
                                showCtx(e, {
                                  kind: "camera",
                                  x: 0,
                                  y: 0,
                                  camera: c,
                                })
                              }
                            >
                              {c.name}
                              {c.online === true && (
                                <span className="badge run" style={{ marginLeft: 6 }}>
                                  在线
                                </span>
                              )}
                            </button>
                          ))}
                      </div>
                    );
                  })}
              </div>
            );
          })}

          {!tree.mines.length && (
            <p className="muted" style={{ padding: "8px 4px" }}>
              暂无煤矿。可点右上角「新增煤矿」，或在树上右键。
            </p>
          )}
        </aside>

        {treeCollapsed && (
          <button
            type="button"
            className="tree-expand-rail"
            title="展开组织树"
            aria-label="展开组织树"
            onClick={() => setTreeCollapsed(false)}
          >
            »
          </button>
        )}

        <div className="camera-main">
          <div className="card camera-list-card">
            <div className="page-head" style={{ marginBottom: 12 }}>
              <h2 className="section-title" style={{ margin: 0 }}>
                {selectedTitle}
                <span className="muted" style={{ fontWeight: 400, marginLeft: 8 }}>
                  共 {selectedCameras.length} 路
                </span>
              </h2>
              {sel?.type === "site" && (
                <button
                  className="btn primary"
                  type="button"
                  onClick={() =>
                    openCameraDialog("create", { siteId: sel.siteId })
                  }
                >
                  新增摄像头
                </button>
              )}
            </div>
            <div className="camera-table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>ID</th>
                    <th>名称</th>
                    <th>摄像仪编码</th>
                    <th>位置描述</th>
                    <th>分析类型</th>
                    <th>接入</th>
                    <th>ZLM 流</th>
                    <th>模板</th>
                    <th>在线</th>
                    <th>启用</th>
                    <th>操作</th>
                  </tr>
                </thead>
                <tbody>
                  {pagedCameras.map((c: any) => (
                    <tr
                      key={c.id}
                      className="row-clickable"
                      onContextMenu={(e) =>
                        showCtx(e, { kind: "camera", x: 0, y: 0, camera: c })
                      }
                      onDoubleClick={() =>
                        openCameraDialog("edit", { camera: c })
                      }
                    >
                      <td>{c.id}</td>
                      <td>{c.name}</td>
                      <td className="mono">{c.camera_code || "—"}</td>
                      <td className="muted" title={c.position_desc || ""}>
                        {c.position_desc || "—"}
                      </td>
                      <td>{analysisTypeLabel(c.analysis_type)}</td>
                      <td>
                        <span className="badge">
                          {c.ingest_mode === "proxy" ? "拉流代理" : "推流登记"}
                        </span>
                      </td>
                      <td className="mono" title={c.in_url || ""}>
                        {c.zlm_app && c.zlm_stream
                          ? `${c.zlm_app}/${c.zlm_stream}`
                          : c.in_url || "—"}
                      </td>
                      <td>
                        <span
                          className={`badge ${
                            c.has_basic_image ? "run" : "stop"
                          }`}
                        >
                          {c.has_basic_image ? "已截取" : "无"}
                        </span>
                      </td>
                      <td>
                        <OnlineBadge cam={c} />
                      </td>
                      <td>
                        <span className={`badge ${c.enabled ? "run" : "stop"}`}>
                          {c.enabled ? "启用" : "禁用"}
                        </span>
                      </td>
                      <td className="row-actions" onDoubleClick={(e) => e.stopPropagation()}>
                        <button
                          className="btn primary"
                          type="button"
                          disabled={!c.flv_url}
                          title={
                            c.flv_url
                              ? c.flv_url
                              : "无可用 FLV 地址（需配置可解析的 ZLM 流）"
                          }
                          onClick={(e) => {
                            e.stopPropagation();
                            setPlaying(c);
                          }}
                        >
                          直播
                        </button>
                      </td>
                    </tr>
                  ))}
                  {!pagedCameras.length && (
                    <tr>
                      <td colSpan={11} className="muted">
                        {sel
                          ? "该节点下暂无摄像头（可在地点上右键「新增摄像头」）"
                          : "请在左侧选择煤矿或地点"}
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
            <Pager
              page={safePage}
              pageSize={PAGE_SIZE}
              total={selectedCameras.length}
              onChange={setPage}
            />
          </div>
        </div>
      </div>

      {ctx && (
        <div
          className="ctx-menu"
          style={{ left: ctx.x, top: ctx.y }}
          onClick={(e) => e.stopPropagation()}
        >
          {ctx.kind === "blank" && (
            <button type="button" onClick={() => openMineDialog("create")}>
              新增煤矿
            </button>
          )}
          {ctx.kind === "mine" && (
            <>
              <button
                type="button"
                onClick={() => openSiteDialog("create", ctx.mine.id)}
              >
                新增地点
              </button>
              <button
                type="button"
                onClick={() => openMineDialog("edit", ctx.mine)}
              >
                编辑
              </button>
              <button
                type="button"
                className="danger"
                onClick={() => deleteMine(ctx.mine)}
              >
                删除
              </button>
            </>
          )}
          {ctx.kind === "site" && (
            <>
              <button
                type="button"
                onClick={() =>
                  openCameraDialog("create", { siteId: ctx.site.id })
                }
              >
                新增摄像头
              </button>
              <button
                type="button"
                onClick={() => openSiteDialog("edit", ctx.mineId, ctx.site)}
              >
                编辑
              </button>
              <button
                type="button"
                className="danger"
                onClick={() => deleteSite(ctx.site, ctx.mineId)}
              >
                删除
              </button>
            </>
          )}
          {ctx.kind === "camera" && (
            <>
              <button
                type="button"
                onClick={() =>
                  openCameraDialog("edit", { camera: ctx.camera })
                }
              >
                编辑
              </button>
              <button
                type="button"
                disabled={!ctx.camera.flv_url}
                onClick={() => {
                  setPlaying(ctx.camera);
                  setCtx(null);
                }}
              >
                直播
              </button>
              <button
                type="button"
                className="danger"
                onClick={() => deleteCamera(ctx.camera)}
              >
                删除
              </button>
            </>
          )}
        </div>
      )}

      {dialog?.type === "mine" && (
        <FormDialog
          title={dialog.mode === "edit" ? "编辑煤矿" : "新增煤矿"}
          onClose={() => setDialog(null)}
          onSubmit={saveMine}
          submitText="保存"
        >
          <div className="form-grid">
            <label>
              煤矿名称
              <input
                required
                value={mineDraft.name}
                onChange={(e) =>
                  setMineDraft({ ...mineDraft, name: e.target.value })
                }
              />
            </label>
            <label>
              煤矿编码
              <input
                placeholder="12位数字"
                value={mineDraft.code}
                onChange={(e) =>
                  setMineDraft({ ...mineDraft, code: e.target.value })
                }
              />
            </label>
            <label>
              启用
              <select
                value={mineDraft.enabled ? "1" : "0"}
                onChange={(e) =>
                  setMineDraft({
                    ...mineDraft,
                    enabled: e.target.value === "1",
                  })
                }
              >
                <option value="1">启用</option>
                <option value="0">禁用</option>
              </select>
            </label>
            <label className="full">
              备注
              <input
                value={mineDraft.remark}
                onChange={(e) =>
                  setMineDraft({ ...mineDraft, remark: e.target.value })
                }
              />
            </label>
          </div>
        </FormDialog>
      )}

      {dialog?.type === "site" && (
        <FormDialog
          title={dialog.mode === "edit" ? "编辑地点" : "新增地点"}
          onClose={() => setDialog(null)}
          onSubmit={saveSite}
          submitText="保存"
        >
          <div className="form-grid">
            <label>
              地点名称
              <input
                required
                placeholder="如：井口闸机"
                value={siteDraft.name}
                onChange={(e) =>
                  setSiteDraft({ ...siteDraft, name: e.target.value })
                }
              />
            </label>
            <label>
              地点编码
              <input
                value={siteDraft.code}
                onChange={(e) =>
                  setSiteDraft({ ...siteDraft, code: e.target.value })
                }
              />
            </label>
            <label>
              启用
              <select
                value={siteDraft.enabled ? "1" : "0"}
                onChange={(e) =>
                  setSiteDraft({
                    ...siteDraft,
                    enabled: e.target.value === "1",
                  })
                }
              >
                <option value="1">启用</option>
                <option value="0">禁用</option>
              </select>
            </label>
            <label className="full">
              备注
              <input
                value={siteDraft.remark}
                onChange={(e) =>
                  setSiteDraft({ ...siteDraft, remark: e.target.value })
                }
              />
            </label>
          </div>
        </FormDialog>
      )}

      {dialog?.type === "camera" && (
        <FormDialog
          title={dialog.mode === "edit" ? "编辑摄像头" : "新增摄像头"}
          onClose={() => setDialog(null)}
          onSubmit={saveCamera}
          submitText="保存"
          width="min(760px, 100%)"
        >
          <div className="form-grid">
            <label>
              名称
              <input
                required
                value={camDraft.name}
                onChange={(e) =>
                  setCamDraft({ ...camDraft, name: e.target.value })
                }
              />
            </label>
            <label>
              归属地点
              <select
                required={dialog.mode === "create"}
                value={camDraft.site_id || ""}
                onChange={(e) =>
                  setCamDraft({
                    ...camDraft,
                    site_id: e.target.value ? Number(e.target.value) : null,
                  })
                }
              >
                <option value="">请选择</option>
                {siteOptions.map((o) => (
                  <option key={o.id} value={o.id}>
                    {o.label}
                  </option>
                ))}
              </select>
            </label>
            <label>
              摄像仪编码 *
              <input
                required
                placeholder="如 34020000001320000001"
                value={camDraft.camera_code}
                onChange={(e) =>
                  setCamDraft({ ...camDraft, camera_code: e.target.value })
                }
              />
            </label>
            <label>
              煤矿编码（可空）
              <input
                placeholder="12 位数字，空则用服务端默认"
                value={camDraft.mine_code}
                onChange={(e) =>
                  setCamDraft({ ...camDraft, mine_code: e.target.value })
                }
              />
            </label>
            <label>
              安装位置分类编码 *
              <input
                required
                placeholder="如 0101（参照 MT/T 1201.6-2023）"
                value={camDraft.position_type}
                onChange={(e) =>
                  setCamDraft({ ...camDraft, position_type: e.target.value })
                }
              />
            </label>
            <label>
              分析类型 *
              <select
                required
                value={camDraft.analysis_type}
                onChange={(e) =>
                  setCamDraft({
                    ...camDraft,
                    analysis_type: e.target.value as "01" | "02",
                  })
                }
              >
                <option value="01">01 人员计数（入）</option>
                <option value="02">02 人员计数（出）</option>
              </select>
            </label>
            <label className="full">
              安装位置描述 *
              <input
                required
                placeholder="如 副井口入井通道"
                value={camDraft.position_desc}
                onChange={(e) =>
                  setCamDraft({ ...camDraft, position_desc: e.target.value })
                }
              />
            </label>
            <label>
              站点编码（可空）
              <input
                placeholder="人员定位分站编码"
                value={camDraft.ps_station_code}
                onChange={(e) =>
                  setCamDraft({ ...camDraft, ps_station_code: e.target.value })
                }
              />
            </label>
            <label>
              数据生成时间 *
              <input
                required
                placeholder="yyyy-MM-dd HH:mm:ss"
                value={camDraft.data_time}
                onChange={(e) =>
                  setCamDraft({ ...camDraft, data_time: e.target.value })
                }
              />
            </label>
            <label className="full">
              接入方式
              <select
                value={camDraft.ingest_mode}
                onChange={(e) =>
                  setCamDraft({
                    ...camDraft,
                    ingest_mode: e.target.value as "push" | "proxy",
                  })
                }
              >
                <option value="push">推流登记（已由 FFmpeg 等推到 ZLM）</option>
                <option value="proxy">拉流代理（摄像头原始地址 → ZLM addStreamProxy）</option>
              </select>
            </label>
            {camDraft.ingest_mode === "proxy" ? (
              <label className="full">
                摄像头原始地址 source_url
                <input
                  required
                  placeholder="rtsp://user:pass@ip:554/Streaming/Channels/101"
                  value={camDraft.source_url}
                  onChange={(e) =>
                    setCamDraft({ ...camDraft, source_url: e.target.value })
                  }
                />
              </label>
            ) : (
              <label className="full">
                ZLM 流地址 in_url（可与下方 app/stream 二选一）
                <input
                  placeholder="rtmp://zlm-host:1935/live/cam001"
                  value={camDraft.in_url}
                  onChange={(e) =>
                    setCamDraft({ ...camDraft, in_url: e.target.value })
                  }
                />
              </label>
            )}
            <label>
              ZLM app
              <input
                placeholder={camDraft.ingest_mode === "proxy" ? "默认 live" : "live"}
                value={camDraft.zlm_app}
                onChange={(e) =>
                  setCamDraft({ ...camDraft, zlm_app: e.target.value })
                }
              />
            </label>
            <label>
              ZLM stream
              <input
                placeholder={
                  camDraft.ingest_mode === "proxy"
                    ? "空则用摄像仪编码生成"
                    : "与 in_url 二选一"
                }
                value={camDraft.zlm_stream}
                onChange={(e) =>
                  setCamDraft({ ...camDraft, zlm_stream: e.target.value })
                }
              />
            </label>
            {dialog.mode === "edit" && camDraft.ingest_mode === "proxy" && (
              <p className="muted full" style={{ margin: 0, gridColumn: "1 / -1" }}>
                平台拉流地址（自动生成）：{camDraft.in_url || "保存后生成"}
              </p>
            )}
            <label className="full">
              基准模板图（可空，保存后自动 ZLM 截图）
              <input
                type="file"
                accept="image/*"
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (!file) return;
                  const reader = new FileReader();
                  reader.onload = () => {
                    const result = String(reader.result || "");
                    setCamDraft({
                      ...camDraft,
                      basic_image_base64: result,
                      has_basic_image: !!result,
                    });
                  };
                  reader.readAsDataURL(file);
                }}
              />
            </label>
            {(camDraft.basic_image_base64 || camDraft.has_basic_image) && (
              <div className="full" style={{ gridColumn: "1 / -1" }}>
                <p className="muted" style={{ margin: "0 0 8px" }}>
                  {camDraft.basic_image_base64
                    ? "当前模板预览（保存时若留空将由服务端自动截取）"
                    : "已有模板（列表未加载大图；保存且未重传时将按流变更策略刷新）"}
                </p>
                {camDraft.basic_image_base64 && (
                  <img
                    src={camDraft.basic_image_base64}
                    alt="基准模板"
                    style={{
                      maxWidth: "100%",
                      maxHeight: 180,
                      borderRadius: 6,
                      border: "1px solid var(--border, #ddd)",
                    }}
                  />
                )}
                {camDraft.basic_image_base64 && (
                  <div className="toolbar" style={{ marginTop: 8 }}>
                    <button
                      className="btn ghost"
                      type="button"
                      onClick={() =>
                        setCamDraft({
                          ...camDraft,
                          basic_image_base64: "",
                          has_basic_image: false,
                        })
                      }
                    >
                      清除本地模板（保存后自动截取）
                    </button>
                  </div>
                )}
              </div>
            )}
            <label>
              启用
              <select
                value={camDraft.enabled ? "1" : "0"}
                onChange={(e) =>
                  setCamDraft({
                    ...camDraft,
                    enabled: e.target.value === "1",
                  })
                }
              >
                <option value="1">启用</option>
                <option value="0">禁用</option>
              </select>
            </label>
            <label>
              备注
              <input
                value={camDraft.remark}
                onChange={(e) =>
                  setCamDraft({ ...camDraft, remark: e.target.value })
                }
              />
            </label>
          </div>
        </FormDialog>
      )}

      {playing && (
        <StreamPlayerModal
          title={`直播 · ${playing.name}`}
          flvUrl={playing.flv_url}
          onClose={() => setPlaying(null)}
        />
      )}
    </div>
  );
}
