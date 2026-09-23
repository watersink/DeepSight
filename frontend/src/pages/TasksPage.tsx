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
  enable_tracking: boolean;
  tracking_algorithm: string;
  /** snapshot 模式 */
  reference_image_url: string;
  check_interval_sec: number;
  shift_threshold_px: number;
  tilt_threshold_deg: number;
  perspective_threshold: number;
  confirm_count: number;
  cooldown_sec: number;
  ssim_skip_threshold: number;
};

type TaskForm = {
  name: string;
  camera_id: number;
  worker_id: string;
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

const TRACKING_ALGORITHM_OPTIONS = [
  { value: "sort", label: "SORT" },
  { value: "bytetrack", label: "ByteTrack" },
  { value: "botsort", label: "BoT-SORT" },
  { value: "ocsort", label: "OC-SORT" },
  { value: "fasttrack", label: "FastTracker" },
  { value: "deepocsort", label: "Deep OC-SORT" },
  { value: "tracktrack", label: "TrackTrack" },
];

function skillRunMode(skills: any[], skillName: string): "stream" | "snapshot" {
  const s = skills.find((x) => x.skill_name === skillName);
  const mode = String(s?.run_mode || "stream").toLowerCase();
  return mode === "snapshot" ? "snapshot" : "stream";
}

/** 需要校准模板的技能（挪移/角度/过暗等），与 run_mode 无关 */
function skillNeedsReferenceTemplate(skills: any[], skillName: string): boolean {
  return !!findFormField(skillFormFields(skills, skillName), "reference_image_url");
}

function skillSupportsTracking(skills: any[], skillName: string) {
  if (skillRunMode(skills, skillName) === "snapshot") return false;
  const s = skills.find((x) => x.skill_name === skillName);
  if (!s) return true;
  const fields = Array.isArray(s.form_fields) ? s.form_fields : [];
  if (fields.some((f: any) => f?.key === "enable_default_sort_tracking")) return true;
  const params = Array.isArray(s.params) ? s.params : [];
  return params.some((p: any) => p?.key === "enable_default_sort_tracking");
}

function trackingDefaults(skills: any[], skillName: string, extra?: any) {
  const fields = skillFormFields(skills, skillName);
  const enableField = findFormField(fields, "enable_default_sort_tracking");
  const algoField = findFormField(fields, "tracking_algorithm");
  const options = algoField?.options?.length
    ? algoField.options
    : TRACKING_ALGORITHM_OPTIONS;
  const defaultEnable =
    enableField == null
      ? true
      : enableField.default !== false &&
        enableField.default !== "0" &&
        enableField.default !== "false";
  const defaultAlgo = String(algoField?.default || options[0]?.value || "sort");
  const savedEnable = extra?.enable_default_sort_tracking;
  const savedAlgo = extra?.tracking_algorithm;
  return {
    enable_tracking: savedEnable == null ? defaultEnable : !!savedEnable,
    tracking_algorithm: String(savedAlgo || defaultAlgo),
    algorithmOptions: options as { value: string; label: string }[],
  };
}

function emptySkill(skillName = "person_presence_detector26"): SkillBinding {
  return {
    key: nextSkillKey(),
    skill_name: skillName,
    gate_direction: "IN",
    enter_count: 0,
    count_line_text: "",
    bypass_line_text: "",
    algorithm_config_id: null,
    enable_tracking: true,
    tracking_algorithm: "sort",
    reference_image_url: "",
    check_interval_sec: 5,
    shift_threshold_px: 40,
    tilt_threshold_deg: 8,
    perspective_threshold: 0.12,
    confirm_count: 3,
    cooldown_sec: 60,
    ssim_skip_threshold: 0.92,
  };
}

const emptyForm = (): TaskForm => ({
  name: "",
  camera_id: 0,
  worker_id: "local",
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

function skillFormFields(skills: any[], skillName: string): any[] {
  const s = skills.find((x) => x.skill_name === skillName);
  const fields = Array.isArray(s?.form_fields) ? s.form_fields : [];
  if (fields.length) return fields;
  // 兼容旧 Worker / 未声明 form_fields 的计数技能
  if (skillName === "person_count_detector26") {
    return [
      {
        key: "gate_direction",
        label: "方向",
        type: "select",
        required: true,
        options: [
          { value: "IN", label: "IN" },
          { value: "OUT", label: "OUT" },
        ],
      },
      { key: "enter_count", label: "计数初始值", type: "number", required: false },
      {
        key: "count_line",
        label: "过线计数线",
        type: "line_draw",
        draw_field: "count_line",
        required: true,
      },
      {
        key: "bypass_line",
        label: "绕行线",
        type: "line_draw",
        draw_field: "bypass_line",
        required: false,
      },
    ];
  }
  return [];
}

function findFormField(fields: any[], key: string) {
  return fields.find((f) => f?.key === key) || null;
}

function snapshotDefaults(skills: any[], skillName: string, extra?: any) {
  const fields = skillFormFields(skills, skillName);
  const num = (key: string, fallback: number) => {
    const f = findFormField(fields, key);
    const saved = extra?.[key];
    if (saved != null && saved !== "") return Number(saved);
    if (f?.default != null && f.default !== "") return Number(f.default);
    return fallback;
  };
  const text = (key: string, fallback = "") => {
    const f = findFormField(fields, key);
    const saved = extra?.[key];
    if (saved != null) return String(saved);
    if (f?.default != null) return String(f.default);
    return fallback;
  };
  return {
    reference_image_url: text("reference_image_url"),
    check_interval_sec: num("check_interval_sec", 5),
    shift_threshold_px: num("shift_threshold_px", 40),
    tilt_threshold_deg: num("tilt_threshold_deg", 8),
    perspective_threshold: num("perspective_threshold", 0.12),
    confirm_count: num("confirm_count", 3),
    cooldown_sec: num("cooldown_sec", 60),
    ssim_skip_threshold: num("ssim_skip_threshold", 0.92),
  };
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
  const [workers, setWorkers] = useState<any[]>([]);
  const [workerSkills, setWorkerSkills] = useState<any[]>([]);

  const [formOpen, setFormOpen] = useState(false);
  const [editingId, setEditingId] = useState<number | null>(null);
  const [form, setForm] = useState<TaskForm>(emptyForm);
  const [saving, setSaving] = useState(false);

  const [error, setError] = useState("");
  const [msg, setMsg] = useState("");
  const [playing, setPlaying] = useState<any | null>(null);
  const [drawTarget, setDrawTarget] = useState<DrawTarget | null>(null);
  const [uploadingRefKey, setUploadingRefKey] = useState<string | null>(null);

  const algoById = useMemo(() => {
    const map = new Map<number, any>();
    for (const a of algos) map.set(a.id, a);
    return map;
  }, [algos]);

  const selectableSkills = useMemo(() => {
    const base = workerSkills.length ? workerSkills : skills;
    const byName = new Map<string, any>();
    for (const s of base) byName.set(String(s.skill_name), s);
    // snapshot 技能跑在 API 巡检，Worker 列表可能没有，始终并入
    for (const s of skills) {
      if (String(s.run_mode || "").toLowerCase() === "snapshot") {
        byName.set(String(s.skill_name), s);
      }
    }
    return Array.from(byName.values());
  }, [workerSkills, skills]);

  const formHasSnapshot = form.skills.some(
    (s) => skillRunMode(skills, s.skill_name) === "snapshot"
  );
  const formHasStream = form.skills.some(
    (s) => skillRunMode(skills, s.skill_name) !== "snapshot"
  );

  const isEdit = editingId != null;
  const drawSkill = drawTarget
    ? form.skills.find((s) => s.key === drawTarget.skillKey)
    : null;

  const loadWorkerSkills = async (workerId: string) => {
    if (!workerId) {
      setWorkerSkills([]);
      return;
    }
    try {
      const res = await api.getWorkerSkills(workerId);
      setWorkerSkills(res.skills || []);
    } catch {
      setWorkerSkills([]);
    }
  };

  const loadFormOptions = async () => {
    const [cams, algorithms, sk, wk] = await Promise.all([
      api.getCameras(),
      api.getAlgorithms(),
      api.getSkills(),
      api.getWorkers().catch(() => ({ items: [] })),
    ]);
    setCameras(cams.items || []);
    setAlgos(algorithms.items || []);
    setSkills(sk.skills || []);
    setWorkers(wk.items || []);
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

  const uploadReferenceImage = async (skillKey: string, file: File | null) => {
    if (!file) return;
    setUploadingRefKey(skillKey);
    setError("");
    try {
      const res = await api.uploadCalibrationImage(file);
      if (!res?.url) throw new Error("上传成功但未返回地址");
      updateSkill(skillKey, { reference_image_url: res.url });
      setMsg("校准模板已上传");
    } catch (err: any) {
      setError(err.message || String(err));
    } finally {
      setUploadingRefKey(null);
    }
  };

  const addSkill = () => {
    const used = new Set(form.skills.map((s) => s.skill_name));
    const pool = selectableSkills;
    const next =
      pool.find((s) => !used.has(s.skill_name))?.skill_name ||
      pool[0]?.skill_name ||
      "person_presence_detector26";
    const track = trackingDefaults(skills, next);
    const snap = snapshotDefaults(skills, next);
    setForm((f) => ({
      ...f,
      skills: [
        ...f.skills,
        {
          ...emptySkill(next),
          enable_tracking: track.enable_tracking,
          tracking_algorithm: track.tracking_algorithm,
          ...snap,
        },
      ],
    }));
  };

  const removeSkill = (key: string) => {
    setForm((f) => {
      if (f.skills.length <= 1) return f;
      return { ...f, skills: f.skills.filter((s) => s.key !== key) };
    });
  };

  const openCreate = () => {
    const defaultWorker =
      workers.find((w) => w.online !== false)?.id ||
      workers[0]?.id ||
      "local";
    const firstSkill =
      selectableSkills[0]?.skill_name || emptySkill().skill_name;
    const track = trackingDefaults(skills, firstSkill);
    const snap = snapshotDefaults(skills, firstSkill);
    setEditingId(null);
    setForm({
      ...emptyForm(),
      camera_id: cameras[0]?.id || 0,
      worker_id: defaultWorker,
      skills: [
        {
          ...emptySkill(firstSkill),
          enable_tracking: track.enable_tracking,
          tracking_algorithm: track.tracking_algorithm,
          ...snap,
        },
      ],
    });
    setFormOpen(true);
    setError("");
    setMsg("");
    loadWorkerSkills(defaultWorker);
  };

  const openEdit = (t: any) => {
    const algo = algoById.get(t.algorithm_config_id);
    const sch = t.schedule || {};
    const wid = t.worker_id || workers[0]?.id || "local";
    const skillName =
      algo?.skill_name || t.skill_name || selectableSkills[0]?.skill_name || "";
    const track = trackingDefaults(skills, skillName, algo?.extra_params);
    const snap = snapshotDefaults(skills, skillName, algo?.extra_params);
    setEditingId(t.id);
    setForm({
      name: t.name || "",
      camera_id: t.camera_id,
      worker_id: wid,
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
          skill_name: skillName,
          gate_direction: algo?.gate_direction || "IN",
          enter_count: algo?.enter_count ?? 0,
          count_line_text: algo?.count_line
            ? JSON.stringify(algo.count_line, null, 2)
            : "",
          bypass_line_text: algo?.bypass_line
            ? JSON.stringify(algo.bypass_line, null, 2)
            : "",
          algorithm_config_id: t.algorithm_config_id ?? null,
          enable_tracking: track.enable_tracking,
          tracking_algorithm: track.tracking_algorithm,
          ...snap,
        },
      ],
    });
    setFormOpen(true);
    setError("");
    setMsg("");
    loadWorkerSkills(wid);
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
        const fields = skillFormFields(skills, s.skill_name);
        const label = skillLabel(skills, s.skill_name);
        for (const f of fields) {
          if (!f?.required) continue;
          if (f.key === "count_line" && !s.count_line_text.trim()) {
            throw new Error(`技能「${label}」必须配置 ${f.label || "count_line"}`);
          }
          if (f.key === "bypass_line" && !s.bypass_line_text.trim()) {
            throw new Error(`技能「${label}」必须配置 ${f.label || "bypass_line"}`);
          }
          if (f.key === "gate_direction" && !String(s.gate_direction || "").trim()) {
            throw new Error(`技能「${label}」必须配置 ${f.label || "方向"}`);
          }
          if (f.key === "reference_image_url" && !String(s.reference_image_url || "").trim()) {
            throw new Error(`技能「${label}」必须配置 ${f.label || "校准模板"}`);
          }
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

        const algoBody: Record<string, unknown> = {
          name: `${finalTaskName}-算法参数`,
          skill_name: s.skill_name,
          gate_direction: s.gate_direction,
          enter_count: Number(s.enter_count) || 0,
          count_line: parseJson(s.count_line_text, `${label}.count_line`),
          bypass_line: parseJson(s.bypass_line_text, `${label}.bypass_line`),
          enabled: form.enabled,
          remark: form.remark || "",
        };
        if (skillSupportsTracking(skills, s.skill_name)) {
          const prevExtra =
            isCurrentTask && s.algorithm_config_id
              ? algoById.get(s.algorithm_config_id)?.extra_params || {}
              : {};
          algoBody.extra_params = {
            ...prevExtra,
            enable_default_sort_tracking: !!s.enable_tracking,
            tracking_algorithm: s.tracking_algorithm || "sort",
          };
        }
        if (
          skillRunMode(skills, s.skill_name) === "snapshot" ||
          skillNeedsReferenceTemplate(skills, s.skill_name)
        ) {
          const prevExtra =
            (algoBody.extra_params as Record<string, unknown>) ||
            (isCurrentTask && s.algorithm_config_id
              ? algoById.get(s.algorithm_config_id)?.extra_params || {}
              : {});
          algoBody.extra_params = {
            ...prevExtra,
            reference_image_url: String(s.reference_image_url || "").trim(),
            check_interval_sec: Number(s.check_interval_sec) || 5,
            shift_threshold_px: Number(s.shift_threshold_px) || 40,
            tilt_threshold_deg: Number(s.tilt_threshold_deg) || 8,
            perspective_threshold: Number(s.perspective_threshold) || 0.12,
            confirm_count: Number(s.confirm_count) || 3,
            cooldown_sec: Number(s.cooldown_sec) || 60,
            ssim_skip_threshold: Number(s.ssim_skip_threshold) || 0.92,
            enable_default_sort_tracking: false,
          };
        }

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

        const isSnap = skillRunMode(skills, s.skill_name) === "snapshot";
        const taskBody = {
          name: finalTaskName,
          camera_id: Number(form.camera_id),
          worker_id: isSnap ? "local" : form.worker_id || "local",
          algorithm_config_id: Number(algorithmConfigId),
          scene_id: sceneId,
          output_format: form.output_format,
          out_fps: Number(form.out_fps) || 15,
          enabled: form.enabled,
          alert_image_enabled: form.alert_image_enabled,
          alert_video_enabled: isSnap ? false : form.alert_video_enabled,
          push_annotated_stream: isSnap ? false : form.push_annotated_stream,
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
            同一摄像头多技能共用一路解码（进程内 fan-out）；挪移/角度/过暗等技能在实时视频中识别，仍需配置校准模板。
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
                <th>模式</th>
                <th>Worker</th>
                <th>运行时间段</th>
                <th>运行状态</th>
                <th>操作</th>
              </tr>
            </thead>
            <tbody>
              {items.map((t) => {
                const algo = algoById.get(t.algorithm_config_id);
                const skillName = t.skill_name || algo?.skill_name || "";
                const showGateBrief = !!findFormField(
                  skillFormFields(skills, skillName),
                  "gate_direction"
                );
                const mode =
                  t.run_mode ||
                  skillRunMode(skills, skillName);
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
                        {skillName || "-"}
                        {showGateBrief && algo?.gate_direction
                          ? ` · ${algo.gate_direction}/${algo.enter_count ?? 0}`
                          : ""}
                      </div>
                    </td>
                    <td>
                      <span className={`badge ${mode === "snapshot" ? "level-3" : "run"}`}>
                        {mode === "snapshot" ? "周期截图" : "实时视频"}
                      </span>
                    </td>
                    <td>
                      {mode === "snapshot"
                        ? "本机巡检"
                        : t.worker_name || t.worker_id || "local"}
                      {mode !== "snapshot" && (
                        <div className="muted mono" style={{ fontSize: 12 }}>
                          {t.worker_id || "local"}
                        </div>
                      )}
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
                  <td colSpan={8} className="muted">
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
                  {formHasStream && (
                    <label>
                      算力 Worker
                      <select
                        required
                        value={form.worker_id || ""}
                        onChange={(e) => {
                          const wid = e.target.value;
                          setForm({ ...form, worker_id: wid });
                          loadWorkerSkills(wid);
                        }}
                      >
                        {!workers.length && (
                          <option value="local">本机 (local)</option>
                        )}
                        {workers.map((w) => (
                          <option key={w.id} value={w.id}>
                            {w.name || w.id}
                            {w.online === false ? " · 离线" : ""}
                            {w.local ? " · 本机" : ""}
                          </option>
                        ))}
                      </select>
                    </label>
                  )}
                  {formHasStream && (
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
                  )}
                  {formHasStream && (
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
                  )}
                  {formHasSnapshot && !formHasStream && (
                    <label className="full">
                      运行方式
                      <input value="周期截图巡检（本机定时 ZLM 截图）" disabled />
                    </label>
                  )}
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
                  {formHasStream && (
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
                  )}
                  {formHasStream && (
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
                  )}
                  <label className="full">
                    备注
                    <input
                      value={form.remark}
                      onChange={(e) => setForm({ ...form, remark: e.target.value })}
                    />
                  </label>
                </div>

                {formHasStream && (
                  <p className="muted" style={{ margin: "0 0 12px", fontSize: 12 }}>
                    关闭「AI 识别结果推流」时不启动 FFmpeg 画框推流，FLV 预览不可用；关闭「报警视频」时不截取证据视频。
                  </p>
                )}
                {formHasSnapshot && (
                  <p className="muted" style={{ margin: "0 0 12px", fontSize: 12 }}>
                    周期截图技能由本机定时 ZLM 截图识别，不占用实时解码 Worker；请预先提供校准模板图片地址。
                  </p>
                )}
                {form.skills.some((s) =>
                  skillNeedsReferenceTemplate(skills, s.skill_name)
                ) &&
                  !formHasSnapshot && (
                  <p className="muted" style={{ margin: "0 0 12px", fontSize: 12 }}>
                    挪移/角度/过暗等技能走实时视频 Worker；请预先提供校准模板图片地址。
                  </p>
                )}

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
                  const fields = skillFormFields(skills, s.skill_name);
                  const mode = skillRunMode(skills, s.skill_name);
                  const gateField = findFormField(fields, "gate_direction");
                  const enterField = findFormField(fields, "enter_count");
                  const countLineField = findFormField(fields, "count_line");
                  const bypassLineField = findFormField(fields, "bypass_line");
                  const canRemove = form.skills.length > 1 && !(isEdit && idx === 0);
                  return (
                    <div className="skill-binding" key={s.key}>
                      <div className="skill-binding-head">
                        <strong>
                          {isEdit && idx === 0 ? "当前技能" : `技能 ${idx + 1}`}
                          <span
                            className={`badge ${mode === "snapshot" ? "level-3" : "run"}`}
                            style={{ marginLeft: 8 }}
                          >
                            {mode === "snapshot" ? "周期截图" : "实时视频"}
                          </span>
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
                            onChange={(e) => {
                              const skill_name = e.target.value;
                              const defaults = trackingDefaults(skills, skill_name);
                              const snap = snapshotDefaults(skills, skill_name);
                              updateSkill(s.key, {
                                skill_name,
                                enable_tracking: defaults.enable_tracking,
                                tracking_algorithm: defaults.tracking_algorithm,
                                ...snap,
                              });
                            }}
                          >
                            {selectableSkills.map((opt) => (
                              <option key={opt.skill_name} value={opt.skill_name}>
                                {opt.name_zh || opt.skill_name}
                                {String(opt.run_mode || "") === "snapshot"
                                  ? " · 截图"
                                  : skillNeedsReferenceTemplate(skills, opt.skill_name)
                                    ? " · 实时+模板"
                                    : ""}
                              </option>
                            ))}
                            {!selectableSkills.length && (
                              <option value={s.skill_name}>{s.skill_name}</option>
                            )}
                          </select>
                        </label>
                        {(mode === "snapshot" ||
                          skillNeedsReferenceTemplate(skills, s.skill_name)) && (
                          <>
                            <div className="form-field full">
                              校准模板图片 *
                              <div className="ref-image-row">
                                <input
                                  required
                                  value={s.reference_image_url}
                                  onChange={(e) =>
                                    updateSkill(s.key, {
                                      reference_image_url: e.target.value,
                                    })
                                  }
                                  placeholder="填写图片 URL / 服务器路径，或右侧上传本地图片"
                                />
                                <label
                                  className={`btn ${
                                    uploadingRefKey === s.key ? "" : "primary"
                                  }`}
                                  style={{
                                    margin: 0,
                                    whiteSpace: "nowrap",
                                    cursor:
                                      uploadingRefKey === s.key
                                        ? "not-allowed"
                                        : "pointer",
                                    opacity: uploadingRefKey === s.key ? 0.7 : 1,
                                  }}
                                >
                                  {uploadingRefKey === s.key
                                    ? "上传中…"
                                    : "上传图片"}
                                  <input
                                    type="file"
                                    accept="image/png,image/jpeg,image/webp,image/bmp"
                                    hidden
                                    disabled={uploadingRefKey === s.key}
                                    onChange={(e) => {
                                      const f = e.target.files?.[0] || null;
                                      e.target.value = "";
                                      void uploadReferenceImage(s.key, f);
                                    }}
                                  />
                                </label>
                              </div>
                              {!!s.reference_image_url.trim() &&
                                /^https?:\/\//i.test(s.reference_image_url.trim()) && (
                                  <div className="ref-image-preview">
                                    <img
                                      src={s.reference_image_url.trim()}
                                      alt="校准模板预览"
                                      onError={(e) => {
                                        (
                                          e.currentTarget as HTMLImageElement
                                        ).style.display = "none";
                                      }}
                                    />
                                  </div>
                                )}
                              <span
                                className="muted"
                                style={{ marginTop: 0, fontSize: 12 }}
                              >
                                支持直接填写地址，或上传本地 png / jpg / webp / bmp（≤8MB）
                              </span>
                            </div>
                            {findFormField(fields, "shift_threshold_px") && (
                              <label>
                                平移告警阈值（像素）
                                <input
                                  type="number"
                                  min={1}
                                  value={s.shift_threshold_px}
                                  onChange={(e) =>
                                    updateSkill(s.key, {
                                      shift_threshold_px: Number(e.target.value),
                                    })
                                  }
                                />
                              </label>
                            )}
                            {findFormField(fields, "tilt_threshold_deg") && (
                              <label>
                                转角告警阈值（度）
                                <input
                                  type="number"
                                  min={0.1}
                                  step={0.1}
                                  value={s.tilt_threshold_deg}
                                  onChange={(e) =>
                                    updateSkill(s.key, {
                                      tilt_threshold_deg: Number(e.target.value),
                                    })
                                  }
                                />
                              </label>
                            )}
                            {findFormField(fields, "perspective_threshold") && (
                              <label>
                                透视告警阈值
                                <input
                                  type="number"
                                  min={0}
                                  step={0.01}
                                  value={s.perspective_threshold}
                                  onChange={(e) =>
                                    updateSkill(s.key, {
                                      perspective_threshold: Number(e.target.value),
                                    })
                                  }
                                />
                              </label>
                            )}
                            <label>
                              连续确认次数
                              <input
                                type="number"
                                min={1}
                                value={s.confirm_count}
                                onChange={(e) =>
                                  updateSkill(s.key, {
                                    confirm_count: Number(e.target.value),
                                  })
                                }
                              />
                            </label>
                            <label>
                              告警冷却（秒）
                              <input
                                type="number"
                                min={0}
                                value={s.cooldown_sec}
                                onChange={(e) =>
                                  updateSkill(s.key, {
                                    cooldown_sec: Number(e.target.value),
                                  })
                                }
                              />
                            </label>
                            {findFormField(fields, "ssim_skip_threshold") && (
                            <label>
                              SSIM 跳过阈值
                              <input
                                type="number"
                                min={0}
                                max={1}
                                step={0.01}
                                value={s.ssim_skip_threshold}
                                onChange={(e) =>
                                  updateSkill(s.key, {
                                    ssim_skip_threshold: Number(e.target.value),
                                  })
                                }
                              />
                            </label>
                            )}
                          </>
                        )}
                        {gateField && (
                          <label>
                            {gateField.label || "方向"}
                            {gateField.required ? " *" : ""}
                            <select
                              value={s.gate_direction}
                              onChange={(e) =>
                                updateSkill(s.key, {
                                  gate_direction: e.target.value,
                                })
                              }
                            >
                              {(gateField.options?.length
                                ? gateField.options
                                : [
                                    { value: "IN", label: "IN" },
                                    { value: "OUT", label: "OUT" },
                                  ]
                              ).map((opt: any) => (
                                <option key={opt.value} value={opt.value}>
                                  {opt.label || opt.value}
                                </option>
                              ))}
                            </select>
                          </label>
                        )}
                        {enterField && (
                          <label>
                            {enterField.label || "计数初始值"}
                            {enterField.required ? " *" : ""}
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
                        )}
                      </div>

                      {(countLineField || bypassLineField) && (
                        <div className="form-grid" style={{ marginTop: 12 }}>
                          {countLineField && (
                            <label className="full">
                              {countLineField.label || "count_line"}
                              {countLineField.required ? "（必填）" : "（可选）"}
                              {countLineField.hint ? (
                                <span className="muted"> · {countLineField.hint}</span>
                              ) : null}
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
                          )}
                          {bypassLineField && (
                            <label className="full">
                              {bypassLineField.label || "bypass_line"}
                              {bypassLineField.required ? "（必填）" : "（可选）"}
                              {bypassLineField.hint ? (
                                <span className="muted"> · {bypassLineField.hint}</span>
                              ) : null}
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
                          )}
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
