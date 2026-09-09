import { FormEvent, useEffect, useMemo, useState } from "react";
import { api } from "../api";
import Pager from "../components/Pager";
import StreamPlayerModal from "../components/StreamPlayerModal";
import SnapLineAnnotator from "../components/SnapLineAnnotator";

const PAGE_SIZE = 10;

const WEEKDAYS: { value: number; label: string }[] = [
  { value: 1, label: "一" },
  { value: 2, label: "二" },
  { value: 3, label: "三" },
  { value: 4, label: "四" },
  { value: 5, label: "五" },
  { value: 6, label: "六" },
  { value: 7, label: "日" },
];

type TaskScheduleForm = {
  enabled: boolean;
  timezone: string;
  days: number[];
  start_time: string;
  end_time: string;
};

type SkillBinding = {
  key: string;
  skill_name: string;
  gate_direction: string;
  enter_count: number;
  count_line_text: string;
  bypass_line_text: string;
  algorithm_config_id: number | null;
};

type TaskForm = {
  name: string;
  camera_id: number;
  scene_id: string;
  output_format: string;
  out_fps: number;
  enabled: boolean;
  alert_image_enabled: boolean;
  alert_video_enabled: boolean;
  push_annotated_stream: boolean;
  remark: string;
  schedule: TaskScheduleForm;
  skills: SkillBinding[];
};

type DrawTarget = {
  field: "count_line" | "bypass_line";
  skillKey: string;
};

let skillKeySeq = 1;
function nextSkillKey() {
  skillKeySeq += 1;
  return `sk-${Date.now()}-${skillKeySeq}`;
}

const defaultSchedule = (): TaskScheduleForm => ({
  enabled: false,
  timezone: "Asia/Shanghai",
  days: [1, 2, 3, 4, 5],
  start_time: "09:00",
  end_time: "18:00",
});

function emptySkill(skillName = "person_presence_detector26"): SkillBinding {
  return {
    key: nextSkillKey(),
    skill_name: skillName,
    gate_direction: "IN",
    enter_count: 0,
    count_line_text: "",
    bypass_line_text: "",
    algorithm_config_id: null,
  };
}

const emptyForm = (): TaskForm => ({
  name: "",
  camera_id: 0,
  scene_id: "scene_001",
  output_format: "rtmp",
  out_fps: 15,
  enabled: true,
  alert_image_enabled: true,
  alert_video_enabled: false,
  push_annotated_stream: false,
  remark: "",
  schedule: defaultSchedule(),
  skills: [emptySkill()],
});

function needsCountLine(skillName: string) {
  return skillName === "person_count_detector26";
}

function normalizeTimeInput(v: string, fallback: string) {
  const t = (v || "").trim();
  if (/^\d{2}:\d{2}/.test(t)) return t.slice(0, 5);
  return fallback;
}

function formatScheduleBrief(schedule: any): string {
  if (!schedule?.enabled) return "手动";
  const days: number[] = Array.isArray(schedule.days) ? schedule.days : [];
  const dayText =
    days.length === 7
      ? "每天"
      : days
          .slice()
          .sort((a, b) => a - b)
          .map((d) => WEEKDAYS.find((w) => w.value === d)?.label || d)
          .join("");
  return `${dayText} ${schedule.start_time || "?"}–${schedule.end_time || "?"}`;
}

function skillLabel(skills: any[], skillName: string) {
  const hit = skills.find((s) => s.skill_name === skillName);
  return hit?.name_zh || skillName;
}

function sceneIdForSkill(base: string, skillName: string, index: number, total: number) {
  const root = (base || "scene").trim() || "scene";
  if (total <= 1) return root;
  const short = skillName
    .replace(/_detector\d*$/i, "")
    .replace(/[^a-zA-Z0-9]+/g, "_")
    .replace(/^_|_$/g, "")
    .slice(0, 24);
  return `${root}_${short || index + 1}`;
}

export default function TasksPage() {
  const [items, setItems] = useState<any[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [cameras, setCameras] = useState<any[]>([]);
  const [algos, setAlgos] = useState<any[]>([]);
  const [skills, setSkills] = useState<any[]>([]);

  const [formOpen, setFormOpen] = useState(false);
  const [editingId, setEditingId] = useState<number | null>(null);
  const [form, setForm] = useState<TaskForm>(emptyForm);
  const [saving, setSaving] = useState(false);

  const [error, setError] = useState("");
  const [msg, setMsg] = useState("");
  const [playing, setPlaying] = useState<any | null>(null);
  const [drawTarget, setDrawTarget] = useState<DrawTarget | null>(null);

  const algoById = useMemo(() => {
    const map = new Map<number, any>();
    for (const a of algos) map.set(a.id, a);
    return map;
  }, [algos]);

  const isEdit = editingId != null;
  const drawSkill = drawTarget
    ? form.skills.find((s) => s.key === drawTarget.skillKey)
    : null;

  const loadFormOptions = async () => {
    const [cams, algorithms, sk] = await Promise.all([
      api.getCameras(),
      api.getAlgorithms(),
      api.getSkills(),
    ]);
    setCameras(cams.items || []);
    setAlgos(algorithms.items || []);
    setSkills(sk.skills || []);
  };

  const loadTasks = async (pageNo = page) => {
    setError("");
    try {
      const tasks = await api.getTasks({ page: pageNo, pageSize: PAGE_SIZE });
      setItems(tasks.items || []);
      setTotal(tasks.meta?.total ?? 0);
      const totalPages = Math.max(
        1,
        Math.ceil((tasks.meta?.total ?? 0) / PAGE_SIZE) || 1
      );
      if (pageNo > totalPages) {
        setPage(totalPages);
      }
    } catch (e: any) {
      setError(e.message || String(e));
    }
  };

  const load = async (pageNo = page) => {
    setError("");
    try {
      await Promise.all([loadFormOptions(), loadTasks(pageNo)]);
    } catch (e: any) {
      setError(e.message || String(e));
    }
  };

  useEffect(() => {
    loadFormOptions().catch((e: any) => setError(e.message || String(e)));
  }, []);

  useEffect(() => {
    loadTasks(page);
  }, [page]);

  const parseJson = (text: string, label: string) => {
    const t = text.trim();
    if (!t) return null;
    try {
      return JSON.parse(t);
    } catch {
      throw new Error(`${label} 不是合法 JSON`);
    }
  };

  const updateSkill = (key: string, patch: Partial<SkillBinding>) => {
    setForm((f) => ({
      ...f,
      skills: f.skills.map((s) => (s.key === key ? { ...s, ...patch } : s)),
    }));
  };

  const addSkill = () => {
    const used = new Set(form.skills.map((s) => s.skill_name));
    const next =
      skills.find((s) => !used.has(s.skill_name))?.skill_name ||
      skills[0]?.skill_name ||
      "person_presence_detector26";
    setForm((f) => ({ ...f, skills: [...f.skills, emptySkill(next)] }));
  };

  const removeSkill = (key: string) => {
    setForm((f) => {
      if (f.skills.length <= 1) return f;
      return { ...f, skills: f.skills.filter((s) => s.key !== key) };
    });
  };

  const openCreate = () => {
    setEditingId(null);
    setForm({
      ...emptyForm(),
      camera_id: cameras[0]?.id || 0,
      skills: [emptySkill(skills[0]?.skill_name || emptySkill().skill_name)],
    });
    setFormOpen(true);
    setError("");
    setMsg("");
  };

  const openEdit = (t: any) => {
    const algo = algoById.get(t.algorithm_config_id);
    const sch = t.schedule || {};
    setEditingId(t.id);
    setForm({
      name: t.name || "",
      camera_id: t.camera_id,
      scene_id: t.scene_id || "scene_001",
      output_format: t.output_format || "rtmp",
      out_fps: t.out_fps || 15,
      enabled: !!t.enabled,
      alert_image_enabled: t.alert_image_enabled !== false,
      alert_video_enabled: !!t.alert_video_enabled,
      push_annotated_stream: !!t.push_annotated_stream,
      remark: t.remark || "",
      schedule: {
        enabled: !!sch.enabled,
        timezone: sch.timezone || "Asia/Shanghai",
        days:
          Array.isArray(sch.days) && sch.days.length
            ? sch.days.map(Number)
            : [1, 2, 3, 4, 5, 6, 7],
        start_time: normalizeTimeInput(sch.start_time, "09:00"),
        end_time: normalizeTimeInput(sch.end_time, "18:00"),
      },
      skills: [
        {
          key: nextSkillKey(),
          skill_name:
            algo?.skill_name || t.skill_name || skills[0]?.skill_name || "",
          gate_direction: algo?.gate_direction || "IN",
          enter_count: algo?.enter_count ?? 0,
          count_line_text: algo?.count_line
            ? JSON.stringify(algo.count_line, null, 2)
            : "",
          bypass_line_text: algo?.bypass_line
            ? JSON.stringify(algo.bypass_line, null, 2)
            : "",
          algorithm_config_id: t.algorithm_config_id ?? null,
        },
      ],
    });
    setFormOpen(true);
    setError("");
    setMsg("");
  };

  const closeForm = () => {
    if (saving) return;
    setFormOpen(false);
    setEditingId(null);
    setDrawTarget(null);
  };

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError("");
    setMsg("");
    setSaving(true);
    try {
      if (!form.skills.length) throw new Error("请至少配置一个技能");
      if (form.schedule.enabled && !form.schedule.days.length) {
        throw new Error("请至少选择一个运行星期");
      }

      const names = form.skills.map((s) => s.skill_name);
      if (new Set(names).size !== names.length) {
        throw new Error("同一摄像头下技能不能重复，请去掉重复项");
      }

      for (const s of form.skills) {
        if (needsCountLine(s.skill_name) && !s.count_line_text.trim()) {
          throw new Error(
            `技能「${skillLabel(skills, s.skill_name)}」必须配置 count_line`
          );
        }
      }

      const schedule = {
        enabled: form.schedule.enabled,
        timezone: form.schedule.timezone || "Asia/Shanghai",
        days: form.schedule.days.length
          ? form.schedule.days
          : [1, 2, 3, 4, 5, 6, 7],
        start_time: normalizeTimeInput(form.schedule.start_time, "09:00"),
        end_time: normalizeTimeInput(form.schedule.end_time, "18:00"),
      };

      let createdExtra = 0;

      for (let i = 0; i < form.skills.length; i++) {
        const s = form.skills[i];
        const label = skillLabel(skills, s.skill_name);
        // 编辑态：仅第一条是当前任务（更新）；其余与新建一样走创建
        const isCurrentTask = isEdit && i === 0;
        const finalTaskName =
          isCurrentTask
            ? form.name
            : form.skills.length > 1 || isEdit
              ? `${form.name}-${label}`
              : form.name;
        const sceneId = isCurrentTask
          ? form.scene_id
          : sceneIdForSkill(
              form.scene_id,
              s.skill_name,
              i,
              form.skills.length
            );

        const algoBody = {
          name: `${finalTaskName}-算法参数`,
          skill_name: s.skill_name,
          gate_direction: s.gate_direction,
          enter_count: Number(s.enter_count) || 0,
          count_line: parseJson(s.count_line_text, `${label}.count_line`),
          bypass_line: parseJson(s.bypass_line_text, `${label}.bypass_line`),
          enabled: form.enabled,
          remark: form.remark || "",
        };

        let algorithmConfigId = s.algorithm_config_id;
        if (isCurrentTask && algorithmConfigId) {
          await api.updateAlgorithm(algorithmConfigId, algoBody);
        } else {
          const created = (await api.createAlgorithm(algoBody)) as any;
          algorithmConfigId = created?.id;
          if (!algorithmConfigId) {
            throw new Error(`创建算法参数失败：${label}`);
          }
        }

        const taskBody = {
          name: finalTaskName,
          camera_id: Number(form.camera_id),
          algorithm_config_id: Number(algorithmConfigId),
          scene_id: sceneId,
          output_format: form.output_format,
          out_fps: Number(form.out_fps) || 15,
          enabled: form.enabled,
          alert_image_enabled: form.alert_image_enabled,
          alert_video_enabled: form.alert_video_enabled,
          push_annotated_stream: form.push_annotated_stream,
          remark: form.remark || "",
          schedule,
        };

        if (isCurrentTask && editingId != null) {
          await api.updateTask(editingId, taskBody);
        } else {
          await api.createTask(taskBody);
          if (isEdit) createdExtra += 1;
        }
      }

      setFormOpen(false);
      setEditingId(null);
      if (isEdit) {
        await load(page);
        setMsg(
          createdExtra > 0
            ? `任务已更新，并新增 ${createdExtra} 个技能任务`
            : "任务已更新"
        );
      } else {
        setPage(1);
        await load(1);
        setMsg(
          form.skills.length > 1
            ? `已为该摄像头创建 ${form.skills.length} 个技能任务`
            : "任务已创建"
        );
      }
    } catch (err: any) {
      setError(err.message || String(err));
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="page-shell">
      <div className="page-head">
        <div>
          <h1>任务配置</h1>
          <p>
            同一摄像头多技能共用一路解码（进程内 fan-out）；列表中仍按技能分条启停。
          </p>
        </div>
        <div className="toolbar">
          <button className="btn primary" type="button" onClick={openCreate}>
            新增任务
          </button>
          <button className="btn" type="button" onClick={() => load(page)}>
            刷新
          </button>
        </div>
      </div>

      {error && !formOpen && <p className="error">{error}</p>}
      {msg && <p className="ok">{msg}</p>}

      <div className="card page-list-card">
        <div className="page-list-scroll">
          <table>
            <thead>
              <tr>
                <th>ID</th>
                <th>名称</th>
                <th>摄像头 / 技能</th>
                <th>运行时间段</th>
                <th>运行状态</th>
                <th>操作</th>
              </tr>
            </thead>
            <tbody>
              {items.map((t) => {
                const algo = algoById.get(t.algorithm_config_id);
                return (
                  <tr key={t.id}>
                    <td>{t.id}</td>
                    <td>
                      {t.name}
                      <div className="muted mono">{t.scene_id}</div>
                    </td>
                    <td>
                      {t.camera_name || `摄像头#${t.camera_id}`}
                      <div className="muted mono">
                        {t.skill_name || algo?.skill_name || "-"}
                        {algo?.gate_direction
                          ? ` · ${algo.gate_direction}/${algo.enter_count ?? 0}`
                          : ""}
                      </div>
                    </td>
                    <td>
                      <div>{formatScheduleBrief(t.schedule)}</div>
                      {t.schedule?.enabled && (
                        <div className="muted" style={{ fontSize: 12 }}>
                          {t.schedule_active ? "当前在窗内" : "当前在窗外"}
                        </div>
                      )}
                    </td>
                    <td>
                      <span
                        className={`badge ${
                          t.runtime_status === "running" ? "run" : "stop"
                        }`}
                      >
                        {t.runtime_status || "未启动"}
                      </span>
                    </td>
                    <td className="row-actions">
                      {t.flv_url && (
                        <button
                          className="btn primary"
                          type="button"
                          title={t.flv_url}
                          onClick={() => setPlaying(t)}
                        >
                          FLV 预览
                        </button>
                      )}
                      <button
                        className="btn primary"
                        type="button"
                        onClick={async () => {
                          setError("");
                          setMsg("");
                          try {
                            await api.startTask(t.id);
                            setMsg(`任务 ${t.name} 已启动`);
                            await loadTasks(page);
                          } catch (err: any) {
                            setError(err.message || String(err));
                          }
                        }}
                      >
                        启动
                      </button>
                      <button
                        className="btn"
                        type="button"
                        onClick={async () => {
                          setError("");
                          try {
                            await api.stopTask(t.id);
                            await loadTasks(page);
                          } catch (err: any) {
                            setError(err.message || String(err));
                          }
                        }}
                      >
                        停止
                      </button>
                      <button className="btn" type="button" onClick={() => openEdit(t)}>
                        编辑
                      </button>
                      <button
                        className="btn danger"
                        type="button"
                        onClick={async () => {
                          if (!confirm(`删除任务 ${t.name}?`)) return;
                          try {
                            const algoId = t.algorithm_config_id;
                            await api.deleteTask(t.id);
                            if (algoId) {
                              try {
                                await api.deleteAlgorithm(algoId);
                                await loadFormOptions();
                              } catch {
                                /* 算法参数可能被其它任务引用，忽略 */
                              }
                            }
                            await loadTasks(page);
                            setMsg("任务已删除");
                          } catch (err: any) {
                            setError(err.message || String(err));
                          }
                        }}
                      >
                        删除
                      </button>
                    </td>
                  </tr>
                );
              })}
              {!items.length && (
                <tr>
                  <td colSpan={6} className="muted">
                    暂无任务。点击右上角「新增任务」开始配置。
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
        <Pager
          page={page}
          pageSize={PAGE_SIZE}
          total={total}
          onChange={setPage}
        />
      </div>

      {formOpen && (
        <div className="modal-backdrop" onClick={closeForm} role="presentation">
          <div
            className="modal-panel task-form-panel"
            onClick={(e) => e.stopPropagation()}
            role="dialog"
            aria-modal="true"
            aria-label={isEdit ? "编辑任务" : "新增任务"}
          >
            <div className="modal-head">
              <div>
                <h2>{isEdit ? "编辑任务" : "新增任务"}</h2>
                <p className="muted">
                  {isEdit
                    ? "可修改当前技能，也可继续添加其它技能（将为同一摄像头新建任务）。"
                    : "同一摄像头可添加多个不同技能，将分别生成任务（scene_id 自动区分）。"}
                </p>
              </div>
              <button className="btn" type="button" onClick={closeForm} disabled={saving}>
                关闭
              </button>
            </div>

            <div className="modal-body">
              {error && <p className="error">{error}</p>}

              <form onSubmit={onSubmit}>
                <h3 className="section-title" style={{ marginTop: 0 }}>
                  任务信息
                </h3>
                <div className="form-grid">
                  <label>
                    任务名称
                    <input
                      required
                      value={form.name}
                      onChange={(e) => setForm({ ...form, name: e.target.value })}
                      placeholder={
                        !isEdit && form.skills.length > 1
                          ? "多技能时自动加后缀，如 名称-过线"
                          : ""
                      }
                    />
                  </label>
                  <label>
                    scene_id{form.skills.length > 1 ? "（当前/前缀）" : ""}
                    <input
                      required
                      value={form.scene_id}
                      onChange={(e) => setForm({ ...form, scene_id: e.target.value })}
                    />
                  </label>
                  <label>
                    摄像头
                    <select
                      required
                      value={form.camera_id || ""}
                      onChange={(e) =>
                        setForm({ ...form, camera_id: Number(e.target.value) })
                      }
                    >
                      {!cameras.length && <option value="">请先添加摄像头</option>}
                      {cameras.map((c) => (
                        <option key={c.id} value={c.id}>
                          #{c.id} {c.name}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label>
                    推流协议
                    <select
                      value={form.output_format}
                      onChange={(e) =>
                        setForm({ ...form, output_format: e.target.value })
                      }
                    >
                      <option value="rtmp">rtmp</option>
                      <option value="rtsp">rtsp</option>
                    </select>
                  </label>
                  <label>
                    out_fps
                    <input
                      type="number"
                      min={1}
                      max={60}
                      value={form.out_fps}
                      onChange={(e) =>
                        setForm({ ...form, out_fps: Number(e.target.value) })
                      }
                    />
                  </label>
                  <label>
                    启用
                    <select
                      value={form.enabled ? "1" : "0"}
                      onChange={(e) =>
                        setForm({ ...form, enabled: e.target.value === "1" })
                      }
                    >
                      <option value="1">启用</option>
                      <option value="0">禁用</option>
                    </select>
                  </label>
                  <label>
                    报警图片
                    <select
                      value={form.alert_image_enabled ? "1" : "0"}
                      onChange={(e) =>
                        setForm({
                          ...form,
                          alert_image_enabled: e.target.value === "1",
                        })
                      }
                    >
                      <option value="1">需要（默认）</option>
                      <option value="0">不需要</option>
                    </select>
                  </label>
                  <label>
                    报警视频
                    <select
                      value={form.alert_video_enabled ? "1" : "0"}
                      onChange={(e) =>
                        setForm({
                          ...form,
                          alert_video_enabled: e.target.value === "1",
                        })
                      }
                    >
                      <option value="0">不需要（默认）</option>
                      <option value="1">需要（截取证据视频）</option>
                    </select>
                  </label>
                  <label>
                    AI 识别结果推流
                    <select
                      value={form.push_annotated_stream ? "1" : "0"}
                      onChange={(e) =>
                        setForm({
                          ...form,
                          push_annotated_stream: e.target.value === "1",
                        })
                      }
                    >
                      <option value="0">不需要（默认）</option>
                      <option value="1">需要（画框后推流）</option>
                    </select>
                  </label>
                  <label className="full">
                    备注
                    <input
                      value={form.remark}
                      onChange={(e) => setForm({ ...form, remark: e.target.value })}
                    />
                  </label>
                </div>

                <p className="muted" style={{ margin: "0 0 12px", fontSize: 12 }}>
                  关闭「AI 识别结果推流」时不启动 FFmpeg 画框推流，FLV 预览不可用；关闭「报警视频」时不截取证据视频。
                </p>

                <h3 className="section-title">运行时间段</h3>
                <p className="muted" style={{ margin: "0 0 10px", fontSize: 12 }}>
                  多技能创建时共用同一时间窗；开启后由调度按窗启停。
                </p>
                <div className="form-grid">
                  <label>
                    按时间段自动运行
                    <select
                      value={form.schedule.enabled ? "1" : "0"}
                      onChange={(e) =>
                        setForm({
                          ...form,
                          schedule: {
                            ...form.schedule,
                            enabled: e.target.value === "1",
                          },
                        })
                      }
                    >
                      <option value="0">关闭（仅手动启停）</option>
                      <option value="1">开启</option>
                    </select>
                  </label>
                  <label>
                    时区
                    <input
                      value={form.schedule.timezone}
                      onChange={(e) =>
                        setForm({
                          ...form,
                          schedule: { ...form.schedule, timezone: e.target.value },
                        })
                      }
                      disabled={!form.schedule.enabled}
                    />
                  </label>
                  <label>
                    开始时间
                    <input
                      type="time"
                      required={form.schedule.enabled}
                      value={form.schedule.start_time}
                      onChange={(e) =>
                        setForm({
                          ...form,
                          schedule: {
                            ...form.schedule,
                            start_time: normalizeTimeInput(e.target.value, "09:00"),
                          },
                        })
                      }
                      disabled={!form.schedule.enabled}
                    />
                  </label>
                  <label>
                    结束时间
                    <input
                      type="time"
                      required={form.schedule.enabled}
                      value={form.schedule.end_time}
                      onChange={(e) =>
                        setForm({
                          ...form,
                          schedule: {
                            ...form.schedule,
                            end_time: normalizeTimeInput(e.target.value, "18:00"),
                          },
                        })
                      }
                      disabled={!form.schedule.enabled}
                    />
                  </label>
                  <label className="full">
                    运行星期
                    <div className="day-toggle-row">
                      {WEEKDAYS.map((d) => {
                        const on = form.schedule.days.includes(d.value);
                        return (
                          <button
                            key={d.value}
                            type="button"
                            className={`day-toggle ${on ? "on" : ""}`}
                            disabled={!form.schedule.enabled}
                            onClick={() => {
                              const days = on
                                ? form.schedule.days.filter((x) => x !== d.value)
                                : [...form.schedule.days, d.value].sort(
                                    (a, b) => a - b
                                  );
                              setForm({
                                ...form,
                                schedule: { ...form.schedule, days },
                              });
                            }}
                          >
                            周{d.label}
                          </button>
                        );
                      })}
                    </div>
                  </label>
                </div>

                <div className="skill-bindings-head">
                  <h3 className="section-title" style={{ margin: 0 }}>
                    识别技能（{form.skills.length}）
                  </h3>
                  <button
                    className="btn"
                    type="button"
                    onClick={addSkill}
                    disabled={
                      !!skills.length && form.skills.length >= skills.length
                    }
                  >
                    添加技能
                  </button>
                </div>
                <p className="muted" style={{ margin: "8px 0 12px", fontSize: 12 }}>
                  {isEdit
                    ? "第一条为当前任务；继续添加的技能保存后会新建任务。"
                    : "每个技能独立参数；保存后列表中会看到多条任务（同一摄像头）。"}
                </p>

                {form.skills.map((s, idx) => {
                  const showLines = needsCountLine(s.skill_name);
                  const canRemove = form.skills.length > 1 && !(isEdit && idx === 0);
                  return (
                    <div className="skill-binding" key={s.key}>
                      <div className="skill-binding-head">
                        <strong>
                          {isEdit && idx === 0 ? "当前技能" : `技能 ${idx + 1}`}
                        </strong>
                        {canRemove && (
                          <button
                            className="btn ghost"
                            type="button"
                            onClick={() => removeSkill(s.key)}
                          >
                            移除
                          </button>
                        )}
                      </div>
                      <div className="form-grid">
                        <label>
                          技能
                          <select
                            value={s.skill_name}
                            onChange={(e) =>
                              updateSkill(s.key, { skill_name: e.target.value })
                            }
                          >
                            {skills.map((opt) => (
                              <option key={opt.skill_name} value={opt.skill_name}>
                                {opt.name_zh || opt.skill_name}
                              </option>
                            ))}
                            {!skills.length && (
                              <option value={s.skill_name}>{s.skill_name}</option>
                            )}
                          </select>
                        </label>
                        <label>
                          闸机方向
                          <select
                            value={s.gate_direction}
                            onChange={(e) =>
                              updateSkill(s.key, {
                                gate_direction: e.target.value,
                              })
                            }
                          >
                            <option value="IN">IN</option>
                            <option value="OUT">OUT</option>
                          </select>
                        </label>
                        <label>
                          enter_count 初值
                          <input
                            type="number"
                            value={s.enter_count}
                            onChange={(e) =>
                              updateSkill(s.key, {
                                enter_count: Number(e.target.value),
                              })
                            }
                          />
                        </label>
                      </div>

                      {showLines && (
                        <div className="form-grid" style={{ marginTop: 12 }}>
                          <label className="full">
                            count_line（过线必填）
                            <div className="toolbar" style={{ margin: "6px 0" }}>
                              <button
                                className="btn primary"
                                type="button"
                                onClick={() =>
                                  setDrawTarget({
                                    field: "count_line",
                                    skillKey: s.key,
                                  })
                                }
                              >
                                截图绘制 count_line
                              </button>
                            </div>
                            <textarea
                              rows={6}
                              placeholder='{"lines":[...],"image_width":1920,"image_height":1080}'
                              value={s.count_line_text}
                              onChange={(e) =>
                                updateSkill(s.key, {
                                  count_line_text: e.target.value,
                                })
                              }
                            />
                          </label>
                          <label className="full">
                            bypass_line（可选）
                            <div className="toolbar" style={{ margin: "6px 0" }}>
                              <button
                                className="btn primary"
                                type="button"
                                onClick={() =>
                                  setDrawTarget({
                                    field: "bypass_line",
                                    skillKey: s.key,
                                  })
                                }
                              >
                                截图绘制 bypass_line
                              </button>
                            </div>
                            <textarea
                              rows={5}
                              value={s.bypass_line_text}
                              onChange={(e) =>
                                updateSkill(s.key, {
                                  bypass_line_text: e.target.value,
                                })
                              }
                            />
                          </label>
                        </div>
                      )}
                    </div>
                  );
                })}

                <div className="toolbar" style={{ marginTop: 16 }}>
                  <button className="btn primary" type="submit" disabled={saving}>
                    {saving
                      ? "保存中…"
                      : isEdit
                        ? form.skills.length > 1
                          ? `保存并新增 ${form.skills.length - 1} 个技能`
                          : "保存修改"
                        : form.skills.length > 1
                          ? `创建 ${form.skills.length} 个技能任务`
                          : "创建任务"}
                  </button>
                  <button
                    className="btn ghost"
                    type="button"
                    onClick={closeForm}
                    disabled={saving}
                  >
                    取消
                  </button>
                </div>
              </form>
            </div>
          </div>
        </div>
      )}

      {playing && (
        <StreamPlayerModal
          title={`预览 · ${playing.name}`}
          flvUrl={playing.flv_url}
          onClose={() => setPlaying(null)}
        />
      )}

      {drawTarget && drawSkill && (
        <SnapLineAnnotator
          title={
            drawTarget.field === "count_line"
              ? `绘制 count_line · ${skillLabel(skills, drawSkill.skill_name)}`
              : `绘制 bypass_line · ${skillLabel(skills, drawSkill.skill_name)}`
          }
          target={drawTarget.field}
          cameras={cameras}
          initialCameraId={form.camera_id || ""}
          initialJson={
            drawTarget.field === "count_line"
              ? drawSkill.count_line_text
              : drawSkill.bypass_line_text
          }
          onClose={() => setDrawTarget(null)}
          onConfirm={(jsonText) => {
            if (drawTarget.field === "count_line") {
              updateSkill(drawTarget.skillKey, { count_line_text: jsonText });
            } else {
              updateSkill(drawTarget.skillKey, { bypass_line_text: jsonText });
            }
            setDrawTarget(null);
            setMsg(`${drawTarget.field} 已从截图标注写入`);
          }}
        />
      )}
    </div>
  );
}
