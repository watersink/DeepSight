import { useEffect, useMemo, useState } from "react";
import { api } from "../api";
import Pager from "../components/Pager";
import coverCount from "../assets/skills/person_count_detector.png";
import coverCount26 from "../assets/skills/person_count_detector26.png";
import coverPresence26 from "../assets/skills/person_presence_detector26.png";

const PAGE_SIZE = 10;
type ViewMode = "list" | "tile";

/** 打包进前端资源，避免依赖 /skills 静态路径是否被后端正确托管 */
const COVER_BY_SKILL: Record<string, string> = {
  person_count_detector: coverCount,
  person_count_detector26: coverCount26,
  person_presence_detector26: coverPresence26,
};

function coverOf(s: any): string {
  const name = String(s?.skill_name || "");
  return (
    COVER_BY_SKILL[name] ||
    s?.cover_image ||
    `/skills/${name}.png`
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
  const [error, setError] = useState("");
  const [page, setPage] = useState(1);
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
      const sk = await api.getSkills();
      setSkills(sk.skills || []);
      setPage(1);
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

  const paged = useMemo(() => {
    const start = (page - 1) * PAGE_SIZE;
    return skills.slice(start, start + PAGE_SIZE);
  }, [skills, page]);

  return (
    <div className="page-shell">
      <div className="page-head">
        <div>
          <h1>算法管理</h1>
          <p>
            展示系统已注册的检测技能（只读）。识别参数在「任务配置 → 新增/编辑任务」中一并填写。
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
                  <th>依赖模型</th>
                  <th>说明</th>
                </tr>
              </thead>
              <tbody>
                {paged.map((s) => (
                  <tr key={s.skill_name}>
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
                    <td className="mono muted">
                      {(s.required_models || []).join(", ") || "—"}
                    </td>
                    <td className="muted">{s.description || "—"}</td>
                  </tr>
                ))}
                {!paged.length && (
                  <tr>
                    <td colSpan={5} className="muted">
                      暂无已注册算法
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          ) : (
            <div className="skill-tile-grid">
              {paged.map((s) => (
                <article key={s.skill_name} className="skill-tile">
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
    </div>
  );
}
