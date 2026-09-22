import { useEffect, useMemo, useState, type ReactNode } from "react";
import { api } from "../api";
import Pager from "../components/Pager";
import coverCount from "../assets/skills/person_count_detector.png";
import coverCount26 from "../assets/skills/person_count_detector26.png";
import coverPresence26 from "../assets/skills/person_presence_detector26.png";
import coverBoarding from "../assets/skills/boarding_detector.png";
import coverNonFixedParking from "../assets/skills/non_fixed_parking_boarding_detector.png";
import coverCameraShift from "../assets/skills/camera_shift_detector.png";
import coverCameraTilt from "../assets/skills/camera_tilt_detector.png";

const PAGE_SIZE = 10;
type ViewMode = "list" | "tile";

/** 打包进前端资源，避免依赖 /skills 静态路径是否被后端正确托管 */
const COVER_BY_SKILL: Record<string, string> = {
  person_count_detector: coverCount,
  person_count_detector26: coverCount26,
  person_presence_detector26: coverPresence26,
  boarding_detector: coverBoarding,
  non_fixed_parking_boarding_detector: coverNonFixedParking,
  camera_shift_detector: coverCameraShift,
  camera_tilt_detector: coverCameraTilt,
};

function coverOf(s: any): string {
  const name = String(s?.skill_name || "");
  return (
    COVER_BY_SKILL[name] ||
    s?.cover_image ||
    `/skills/${name}.png`
  );
}

function formatAlertCodes(codes: any): string {
  if (!codes || typeof codes !== "object") return "";
  return Object.entries(codes)
    .map(([k, v]: [string, any]) => {
      if (v && typeof v === "object") {
        const code = v.code != null ? String(v.code) : "";
        const desc = v.description ? String(v.description) : "";
        return `${k}${code ? `=${code}` : ""}${desc ? ` ${desc}` : ""}`;
      }
      return `${k}=${String(v)}`;
    })
    .join("；");
}

function SkillDetailModal({
  skill,
  workers,
  onClose,
}: {
  skill: any;
  workers: ReactNode;
  onClose: () => void;
}) {
  const params: any[] = skill?.params || [];
  const formFields: any[] = skill?.form_fields || [];
  const alerts: any[] = skill?.alert_definitions || [];

  return (
    <div className="modal-backdrop" onClick={onClose} role="presentation">
      <div
        className="modal-panel skill-detail-panel"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-labelledby="skill-detail-title"
      >
        <div className="modal-head">
          <div>
            <h2 id="skill-detail-title">{skill.name_zh || skill.skill_name}</h2>
            <p className="mono muted">{skill.skill_name}</p>
          </div>
          <button className="btn" type="button" onClick={onClose}>
            关闭
          </button>
        </div>
        <div className="modal-body skill-detail-body">
          <div className="skill-detail-hero">
            <img src={coverOf(skill)} alt="" />
            <div>
              <p className="skill-detail-desc">{skill.description || "暂无说明"}</p>
              <div className="skill-detail-meta">
                <span>类型：{skill.type || "—"}</span>
                <span>版本：{skill.version || "—"}</span>
                <span className="mono">
                  模型：{(skill.required_models || []).join(", ") || "—"}
                </span>
              </div>
              <div className="skill-tile-workers" style={{ marginTop: 8 }}>
                <span className="muted" style={{ fontSize: 12 }}>
                  部署节点
                </span>
                {workers}
              </div>
            </div>
          </div>

          <section>
            <h3>任务配置项</h3>
            {formFields.length ? (
              <table className="skill-kv-table">
                <thead>
                  <tr>
                    <th>参数</th>
                    <th>类型</th>
                    <th>必填</th>
                    <th>默认</th>
                    <th>说明</th>
                  </tr>
                </thead>
                <tbody>
                  {formFields.map((f) => (
                    <tr key={f.key}>
                      <td>{f.label || f.key}</td>
                      <td className="mono muted">{f.type || "—"}</td>
                      <td>{f.required ? "是" : "否"}</td>
                      <td className="mono">
                        {f.default == null || f.default === ""
                          ? "—"
                          : String(f.default)}
                      </td>
                      <td className="muted">{f.hint || "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <p className="muted">该技能无任务级专用表单字段。</p>
            )}
          </section>

          <section>
            <h3>默认检测参数</h3>
            {params.length ? (
              <table className="skill-kv-table">
                <thead>
                  <tr>
                    <th>参数</th>
                    <th>键名</th>
                    <th>默认值</th>
                  </tr>
                </thead>
                <tbody>
                  {params.map((p) => (
                    <tr key={p.key}>
                      <td>{p.label || p.key}</td>
                      <td className="mono muted">{p.key}</td>
                      <td className="mono">{p.value}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <p className="muted">无默认参数。</p>
            )}
          </section>

          <section>
            <h3>告警 / 识别类型</h3>
            {alerts.length ? (
              <table className="skill-kv-table">
                <thead>
                  <tr>
                    <th>项</th>
                    <th>级别</th>
                    <th>说明</th>
                  </tr>
                </thead>
                <tbody>
                  {alerts.map((a, i) => (
                    <tr key={a.key || i}>
                      <td className="mono">{a.key || "—"}</td>
                      <td>{a.level == null ? "—" : a.level}</td>
                      <td>
                        {a.description || "—"}
                        {a.codes ? (
                          <div className="muted" style={{ marginTop: 4 }}>
                            {formatAlertCodes(a.codes)}
                          </div>
                        ) : null}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <p className="muted">无独立告警定义。</p>
            )}
          </section>
        </div>
      </div>
    </div>
  );
}

function ListIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true">
      <path
        fill="currentColor"
        d="M2 3h12v1.5H2V3zm0 4.25h12V8.75H2V7.25zm0 4.25h12V13H2v-1.5z"
      />
    </svg>
  );
}

function TileIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true">
      <path
        fill="currentColor"
        d="M2 2h5v5H2V2zm7 0h5v5H9V2zM2 9h5v5H2V9zm7 0h5v5H9V9z"
      />
    </svg>
  );
}

export default function AlgorithmsPage() {
  const [skills, setSkills] = useState<any[]>([]);
  const [skillWorkers, setSkillWorkers] = useState<
    Record<
      string,
      {
        id: string;
        name: string;
        online: boolean | null;
        triton_url?: string | null;
        models_ready?: boolean;
      }[]
    >
  >({});
  const [error, setError] = useState("");
  const [page, setPage] = useState(1);
  const [detail, setDetail] = useState<any | null>(null);
  const [viewMode, setViewMode] = useState<ViewMode>(() => {
    try {
      const saved = localStorage.getItem("algorithms_view_mode");
      return saved === "tile" || saved === "list" ? saved : "tile";
    } catch {
      return "tile";
    }
  });

  const load = async () => {
    setError("");
    try {
      const [sk, modelsRes, wk] = await Promise.all([
        api.getSkills(),
        api.getModels().catch(() => ({ items: [], workers: [] as any[] })),
        api.getWorkers().catch(() => ({ items: [] as any[] })),
      ]);
      const skillList = sk.skills || [];
      setSkills(skillList);
      setPage(1);

      // model_name -> workers that have it
      const modelWorkers: Record<
        string,
        { id: string; name: string; online: boolean; triton_url?: string | null }[]
      > = {};
      for (const m of modelsRes.items || []) {
        const name = String(m.name || "").trim();
        if (!name) continue;
        modelWorkers[name] = (m.workers || []).map((w: any) => ({
          id: String(w.worker_id),
          name: String(w.worker_name || w.worker_id),
          online: !!w.ready,
          triton_url: w.triton_url || null,
        }));
      }

      const workerOnline: Record<string, boolean | null> = {};
      for (const w of wk.items || []) {
        workerOnline[String(w.id)] =
          typeof w.online === "boolean" ? w.online : null;
      }
      for (const w of modelsRes.workers || []) {
        if (workerOnline[w.worker_id] == null && typeof w.online === "boolean") {
          workerOnline[String(w.worker_id)] = w.online;
        }
      }

      const skillMap: Record<
        string,
        {
          id: string;
          name: string;
          online: boolean | null;
          triton_url?: string | null;
          models_ready?: boolean;
        }[]
      > = {};

      for (const skill of skillList) {
        const skillName = String(skill.skill_name || "").trim();
        if (!skillName) continue;
        const required: string[] = (skill.required_models || [])
          .map((x: any) => String(x || "").trim())
          .filter(Boolean);

        const byId = new Map<
          string,
          {
            id: string;
            name: string;
            online: boolean | null;
            triton_url?: string | null;
            models_ready?: boolean;
          }
        >();

        if (required.length) {
          // 交集：依赖模型都出现在该 Worker 上才算部署
          let first = true;
          const counts: Record<string, number> = {};
          const meta: Record<
            string,
            { name: string; triton_url?: string | null; readyHits: number }
          > = {};
          for (const modelName of required) {
            const nodes = modelWorkers[modelName] || [];
            for (const n of nodes) {
              counts[n.id] = (counts[n.id] || 0) + 1;
              if (!meta[n.id]) {
                meta[n.id] = {
                  name: n.name,
                  triton_url: n.triton_url,
                  readyHits: 0,
                };
              }
              if (n.online) meta[n.id].readyHits += 1;
              if (first && n.triton_url) meta[n.id].triton_url = n.triton_url;
            }
            first = false;
          }
          for (const [id, cnt] of Object.entries(counts)) {
            if (cnt < required.length) continue;
            const info = meta[id];
            byId.set(id, {
              id,
              name: info.name,
              online: workerOnline[id] ?? true,
              triton_url: info.triton_url,
              models_ready: info.readyHits >= required.length,
            });
          }
        }

        // 无依赖模型或 Triton 未扫到时，回退：Worker 技能列表含该 skill
        if (!byId.size) {
          await Promise.all(
            (wk.items || []).map(async (w: any) => {
              const id = String(w.id);
              try {
                const res = await api.getWorkerSkills(id);
                const hit = (res.skills || []).some(
                  (s: any) => String(s.skill_name) === skillName
                );
                if (!hit) return;
                byId.set(id, {
                  id,
                  name: String(w.name || id),
                  online: typeof w.online === "boolean" ? w.online : null,
                  triton_url: w.triton_url || null,
                  models_ready: undefined,
                });
              } catch {
                /* ignore */
              }
            })
          );
        }

        skillMap[skillName] = Array.from(byId.values());
      }

      setSkillWorkers(skillMap);
    } catch (e: any) {
      setError(e.message || String(e));
    }
  };

  useEffect(() => {
    load();
  }, []);

  useEffect(() => {
    try {
      localStorage.setItem("algorithms_view_mode", viewMode);
    } catch {
      /* ignore */
    }
  }, [viewMode]);

  useEffect(() => {
    if (!detail) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setDetail(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [detail]);

  const paged = useMemo(() => {
    const start = (page - 1) * PAGE_SIZE;
    return skills.slice(start, start + PAGE_SIZE);
  }, [skills, page]);

  const renderWorkerTags = (skillName: string) => {
    const nodes = skillWorkers[skillName] || [];
    if (!nodes.length) {
      return <span className="muted">未部署 / 未探测到</span>;
    }
    return (
      <div className="worker-tag-row">
        {nodes.map((n) => (
          <span
            key={n.id}
            className={`badge worker-tag ${
              n.online === false || n.models_ready === false
                ? "stop"
                : n.online || n.models_ready
                  ? "run"
                  : ""
            }`}
            title={
              [
                n.name,
                n.triton_url ? `Triton: ${n.triton_url}` : "",
                n.models_ready === false ? "依赖模型未就绪" : "",
                n.online === false ? "Worker 离线" : "",
              ]
                .filter(Boolean)
                .join(" · ")
            }
          >
            {n.name}
            {n.triton_url ? ` · ${n.triton_url}` : ""}
            {n.online === false ? " · 离线" : ""}
            {n.models_ready === false ? " · 模型未就绪" : ""}
          </span>
        ))}
      </div>
    );
  };

  return (
    <div className="page-shell">
      <div className="page-head">
        <div>
          <h1>算法管理</h1>
          <p>
            展示系统已注册的检测技能（只读）。点击技能可查看默认检测参数与告警说明。
            任务级参数在「任务配置 → 新增/编辑任务」中填写。
            Worker 标签优先按该技能依赖的 Triton 模型所在机器展示（含 Triton 地址）。
          </p>
        </div>
        <div className="toolbar">
          <div className="view-toggle" role="group" aria-label="展示方式">
            <button
              type="button"
              className={`btn view-toggle-btn ${viewMode === "list" ? "active" : ""}`}
              title="列表展示"
              aria-pressed={viewMode === "list"}
              onClick={() => setViewMode("list")}
            >
              <ListIcon />
              <span>列表</span>
            </button>
            <button
              type="button"
              className={`btn view-toggle-btn ${viewMode === "tile" ? "active" : ""}`}
              title="平铺展示"
              aria-pressed={viewMode === "tile"}
              onClick={() => setViewMode("tile")}
            >
              <TileIcon />
              <span>平铺</span>
            </button>
          </div>
          <button className="btn" onClick={load}>
            刷新
          </button>
        </div>
      </div>

      {error && <p className="error">{error}</p>}

      <div className="card page-list-card">
        <div className="page-list-scroll">
          {viewMode === "list" ? (
            <table>
              <thead>
                <tr>
                  <th>封面</th>
                  <th>中文名称</th>
                  <th>skill_name</th>
                  <th>部署 Worker</th>
                  <th>依赖模型</th>
                  <th>说明</th>
                </tr>
              </thead>
              <tbody>
                {paged.map((s) => (
                  <tr
                    key={s.skill_name}
                    className="skill-list-row"
                    onClick={() => setDetail(s)}
                  >
                    <td>
                      <img
                        className="skill-list-thumb"
                        src={coverOf(s)}
                        alt=""
                        width={72}
                        height={45}
                      />
                    </td>
                    <td>{s.name_zh || s.skill_name}</td>
                    <td className="mono">{s.skill_name}</td>
                    <td>{renderWorkerTags(s.skill_name)}</td>
                    <td className="mono muted">
                      {(s.required_models || []).join(", ") || "—"}
                    </td>
                    <td className="muted">{s.description || "—"}</td>
                  </tr>
                ))}
                {!paged.length && (
                  <tr>
                    <td colSpan={6} className="muted">
                      暂无已注册算法
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          ) : (
            <div className="skill-tile-grid">
              {paged.map((s) => (
                <article
                  key={s.skill_name}
                  className="skill-tile"
                  onClick={() => setDetail(s)}
                  role="button"
                  tabIndex={0}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      setDetail(s);
                    }
                  }}
                >
                  <div className="skill-tile-cover">
                    <img
                      src={coverOf(s)}
                      alt={s.name_zh || s.skill_name}
                      loading="eager"
                    />
                  </div>
                  <div className="skill-tile-body">
                    <h3>{s.name_zh || s.skill_name}</h3>
                    <p className="mono muted">{s.skill_name}</p>
                    <div className="skill-tile-workers">
                      <span className="muted" style={{ fontSize: 12 }}>
                        部署节点
                      </span>
                      {renderWorkerTags(s.skill_name)}
                    </div>
                    <p className="skill-tile-desc">{s.description || "暂无说明"}</p>
                    <p className="mono muted skill-tile-models">
                      模型：{(s.required_models || []).join(", ") || "—"}
                    </p>
                  </div>
                </article>
              ))}
              {!paged.length && (
                <p className="muted" style={{ gridColumn: "1 / -1" }}>
                  暂无已注册算法
                </p>
              )}
            </div>
          )}
        </div>
        <Pager
          page={page}
          pageSize={PAGE_SIZE}
          total={skills.length}
          onChange={setPage}
        />
      </div>

      {detail && (
        <SkillDetailModal
          skill={detail}
          workers={renderWorkerTags(detail.skill_name)}
          onClose={() => setDetail(null)}
        />
      )}
    </div>
  );
}
