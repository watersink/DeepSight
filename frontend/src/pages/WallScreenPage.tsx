import { useCallback, useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api";
import { useBranding } from "../BrandContext";
import StreamPlayer from "../components/StreamPlayer";

type Kpi = { label: string; value: number; warn?: boolean };
type SiteBar = { name: string; value: number };
type QualityItem = { key: string; label: string; value: number; color: string };
type AlertRow = {
  id: number;
  time: string;
  place: string;
  type: string;
  level: string;
  thumb?: string | null;
};
type VideoSlot = {
  id: string;
  title: string;
  status: "alarm" | "normal";
  flvUrl?: string | null;
};

const QUALITY_COLORS = [
  "#22d3ee",
  "#38bdf8",
  "#818cf8",
  "#f472b6",
  "#fb923c",
  "#fbbf24",
  "#4ade80",
  "#a78bfa",
  "#94a3b8",
];

const SAMPLE = {
  overview: [
    { label: "摄像头", value: 128 },
    { label: "运行任务", value: 36 },
    { label: "今日报警", value: 17, warn: true },
    { label: "今日事件", value: 240 },
    { label: "未处置", value: 5, warn: true },
  ] as Kpi[],
  enterCount: 86,
  exitCount: 79,
  presenceBySite: [
    { name: "主井口", value: 12 },
    { name: "辅运巷", value: 4 },
    { name: "井底车场", value: 7 },
    { name: "候车室", value: 3 },
  ] as SiteBar[],
  quality: [
    { key: "occlusion", label: "遮挡", value: 18, color: QUALITY_COLORS[0] },
    { key: "shift", label: "挪动", value: 9, color: QUALITY_COLORS[1] },
    { key: "dark", label: "过暗", value: 14, color: QUALITY_COLORS[2] },
    { key: "overexp", label: "过曝", value: 6, color: QUALITY_COLORS[3] },
    { key: "blur", label: "模糊", value: 11, color: QUALITY_COLORS[4] },
    { key: "freeze", label: "冻结", value: 5, color: QUALITY_COLORS[5] },
    { key: "lost", label: "丢失", value: 8, color: QUALITY_COLORS[6] },
    { key: "shake", label: "抖动", value: 7, color: QUALITY_COLORS[7] },
    { key: "resolution", label: "分辨率", value: 4, color: QUALITY_COLORS[8] },
  ] as QualityItem[],
  videos: [
    { id: "v1", title: "主井口-入井", status: "alarm" },
    { id: "v2", title: "辅运巷", status: "normal" },
    { id: "v3", title: "井底车场", status: "normal" },
    { id: "v4", title: "候车室", status: "normal" },
    { id: "v5", title: "轨运大巷", status: "normal" },
    { id: "v6", title: "风门通道", status: "normal" },
  ] as VideoSlot[],
  trend: [2, 3, 2, 4, 5, 3, 6, 8, 7, 9, 11, 14, 12, 10, 8, 13, 16, 12, 9, 7, 5, 4, 3, 2],
  alerts: [
    {
      id: 1,
      time: "15:28",
      place: "主井口",
      type: "非常规通道入井",
      level: "二级",
    },
    {
      id: 2,
      time: "15:21",
      place: "辅运巷",
      type: "画面遮挡",
      level: "二级",
    },
    {
      id: 3,
      time: "15:10",
      place: "井底车场",
      type: "视频丢失",
      level: "二级",
    },
    {
      id: 4,
      time: "14:56",
      place: "候车室",
      type: "图像模糊",
      level: "二级",
    },
    {
      id: 5,
      time: "14:40",
      place: "轨运大巷",
      type: "摄像头挪动",
      level: "二级",
    },
  ] as AlertRow[],
  siteRank: [
    { name: "主井口", value: 9 },
    { name: "辅运巷", value: 6 },
    { name: "井底车场", value: 4 },
    { name: "候车室", value: 3 },
  ] as SiteBar[],
  health: { running: 36, stopped: 4, noVideo: 2 },
};

const TYPE_LABELS: Record<string, string> = {
  "04": "非常规通道入井",
  "05": "非常规通道出井",
  "06": "入井闸机出闸",
  "07": "出井闸机入闸",
  "08": "摄像头位置挪移",
  "09": "摄像头角度偏离",
  dark: "画面过暗",
  overexp: "画面过曝",
  blur: "图像模糊",
  occlusion: "画面遮挡",
  freeze: "画面冻结",
  video_lost: "视频丢失",
  shake: "画面抖动",
  resolution_anomaly: "分辨率异常",
};

function pad2(n: number) {
  return String(n).padStart(2, "0");
}

function formatClock(d: Date) {
  return `${d.getFullYear()}-${pad2(d.getMonth() + 1)}-${pad2(d.getDate())} ${pad2(
    d.getHours()
  )}:${pad2(d.getMinutes())}:${pad2(d.getSeconds())}`;
}

function formatHm(iso: string) {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso.slice(11, 16) || iso;
  return `${pad2(d.getHours())}:${pad2(d.getMinutes())}`;
}

function levelText(level: unknown) {
  const n = Number(level);
  if (!Number.isFinite(n) || n <= 0) return "二级";
  const map = ["", "一级", "二级", "三级", "四级"];
  return map[n] || `${n}级`;
}

function Donut({ items }: { items: QualityItem[] }) {
  const total = items.reduce((s, x) => s + x.value, 0) || 1;
  let acc = 0;
  const stops = items.map((it) => {
    const start = (acc / total) * 100;
    acc += it.value;
    const end = (acc / total) * 100;
    return `${it.color} ${start}% ${end}%`;
  });
  return (
    <div className="wall-donut-wrap">
      <div
        className="wall-donut"
        style={{ background: `conic-gradient(${stops.join(",")})` }}
      >
        <div className="wall-donut-hole">
          <div className="wall-donut-total">{total}</div>
          <div className="wall-donut-sub">合计</div>
        </div>
      </div>
      <ul className="wall-donut-legend">
        {items.map((it) => (
          <li key={it.key}>
            <span className="wall-dot" style={{ background: it.color }} />
            <span>{it.label}</span>
            <b>{it.value}</b>
          </li>
        ))}
      </ul>
    </div>
  );
}

function TrendChart({ values }: { values: number[] }) {
  const w = 640;
  const h = 140;
  const max = Math.max(...values, 1);
  const step = w / Math.max(values.length - 1, 1);
  const pts = values
    .map((v, i) => {
      const x = i * step;
      const y = h - (v / max) * (h - 16) - 8;
      return `${x},${y}`;
    })
    .join(" ");
  const area = `0,${h} ${pts} ${w},${h}`;
  return (
    <svg className="wall-trend" viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none">
      <defs>
        <linearGradient id="wallTrendFill" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#22d3ee" stopOpacity="0.35" />
          <stop offset="100%" stopColor="#22d3ee" stopOpacity="0.02" />
        </linearGradient>
        <linearGradient id="wallTrendStroke" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stopColor="#22d3ee" />
          <stop offset="70%" stopColor="#fb923c" />
          <stop offset="100%" stopColor="#f87171" />
        </linearGradient>
      </defs>
      {[0.25, 0.5, 0.75].map((p) => (
        <line
          key={p}
          x1="0"
          x2={w}
          y1={h * p}
          y2={h * p}
          stroke="rgba(148,163,184,0.15)"
          strokeWidth="1"
        />
      ))}
      <polygon points={area} fill="url(#wallTrendFill)" />
      <polyline
        points={pts}
        fill="none"
        stroke="url(#wallTrendStroke)"
        strokeWidth="3"
        strokeLinejoin="round"
        strokeLinecap="round"
      />
    </svg>
  );
}

function BarList({
  items,
  color = "#22d3ee",
}: {
  items: SiteBar[];
  color?: string;
}) {
  const max = Math.max(...items.map((x) => x.value), 1);
  return (
    <ul className="wall-bars">
      {items.map((it) => (
        <li key={it.name}>
          <div className="wall-bars-label">
            <span>{it.name}</span>
            <b>{it.value}</b>
          </div>
          <div className="wall-bars-track">
            <div
              className="wall-bars-fill"
              style={{ width: `${(it.value / max) * 100}%`, background: color }}
            />
          </div>
        </li>
      ))}
    </ul>
  );
}

export default function WallScreenPage() {
  const { branding } = useBranding();
  const [now, setNow] = useState(() => new Date());
  const [overview, setOverview] = useState(SAMPLE.overview);
  const [enterCount, setEnterCount] = useState(SAMPLE.enterCount);
  const [exitCount, setExitCount] = useState(SAMPLE.exitCount);
  const [presenceBySite, setPresenceBySite] = useState(SAMPLE.presenceBySite);
  const [quality, setQuality] = useState(SAMPLE.quality);
  const [videos, setVideos] = useState(SAMPLE.videos);
  const [trend, setTrend] = useState(SAMPLE.trend);
  const [alerts, setAlerts] = useState(SAMPLE.alerts);
  const [siteRank, setSiteRank] = useState(SAMPLE.siteRank);
  const [health, setHealth] = useState(SAMPLE.health);
  const [liveMode, setLiveMode] = useState(false);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const t = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(t);
  }, []);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const today = new Date();
      const from = new Date(
        today.getFullYear(),
        today.getMonth(),
        today.getDate()
      ).toISOString();

      const [cameras, tasks, alertsToday, eventsToday, alertsNew, alertsRecent] =
        await Promise.all([
          api.getCameras().catch(() => null),
          api.getTasks({ page: 1, pageSize: 200 }).catch(() => null),
          api
            .getAlerts({ timeFrom: from, page: 1, pageSize: 1 })
            .catch(() => null),
          api
            .getEvents({ timeFrom: from, page: 1, pageSize: 1 })
            .catch(() => null),
          api.getAlerts({ status: "new", page: 1, pageSize: 1 }).catch(() => null),
          api.getAlerts({ page: 1, pageSize: 8 }).catch(() => null),
        ]);

      const camTotal = cameras?.meta?.total ?? cameras?.items?.length ?? SAMPLE.overview[0].value;
      const taskItems = tasks?.items || [];
      const running = taskItems.filter((t) => t.runtime_status === "running").length;
      const stopped = taskItems.filter((t) => t.runtime_status !== "running").length;
      const noVideo = taskItems.filter(
        (t) => t.runtime_status === "running" && !t.flv_url
      ).length;
      const alertTotal = alertsToday?.meta?.total ?? SAMPLE.overview[2].value;
      const eventTotal = eventsToday?.meta?.total ?? SAMPLE.overview[3].value;
      const pending = alertsNew?.meta?.total ?? SAMPLE.overview[4].value;

      setOverview([
        { label: "摄像头", value: camTotal },
        { label: "运行任务", value: running || SAMPLE.overview[1].value },
        { label: "今日报警", value: alertTotal, warn: true },
        { label: "今日事件", value: eventTotal },
        { label: "未处置", value: pending, warn: true },
      ]);
      setHealth({
        running: running || SAMPLE.health.running,
        stopped: stopped || SAMPLE.health.stopped,
        noVideo: noVideo || SAMPLE.health.noVideo,
      });

      const recent = alertsRecent?.items || [];
      if (recent.length) {
        const rows: AlertRow[] = recent.slice(0, 5).map((a) => {
          const types = Array.isArray(a.recognition_types) ? a.recognition_types : [];
          const code = String(types[0] || "");
          return {
            id: a.id,
            time: formatHm(String(a.created_at || "")),
            place: String(a.scene_id || a.payload?.camera_name || "未知点位"),
            type: TYPE_LABELS[code] || a.message || code || "报警",
            level: levelText(a.alarm_level),
            thumb: a.image_url || a.payload?.image_minio_url || null,
          };
        });
        setAlerts(rows);

        const rankMap = new Map<string, number>();
        for (const a of recent) {
          const key = String(a.scene_id || "未知");
          rankMap.set(key, (rankMap.get(key) || 0) + 1);
        }
        const rank = [...rankMap.entries()]
          .map(([name, value]) => ({ name, value }))
          .sort((a, b) => b.value - a.value)
          .slice(0, 4);
        if (rank.length) setSiteRank(rank);

        const qMap = new Map<string, number>();
        for (const a of recent) {
          for (const t of a.recognition_types || []) {
            const k = String(t);
            if (
              [
                "dark",
                "overexp",
                "blur",
                "occlusion",
                "freeze",
                "video_lost",
                "shake",
                "resolution_anomaly",
                "08",
                "09",
              ].includes(k)
            ) {
              qMap.set(k, (qMap.get(k) || 0) + 1);
            }
          }
        }
        if (qMap.size) {
          const labelOf = (k: string) => TYPE_LABELS[k] || k;
          const nextQ = SAMPLE.quality.map((base, i) => {
            const aliases: Record<string, string[]> = {
              occlusion: ["occlusion"],
              shift: ["08", "09"],
              dark: ["dark"],
              overexp: ["overexp"],
              blur: ["blur"],
              freeze: ["freeze"],
              lost: ["video_lost"],
              shake: ["shake"],
              resolution: ["resolution_anomaly"],
            };
            const keys = aliases[base.key] || [];
            const value = keys.reduce((s, k) => s + (qMap.get(k) || 0), 0);
            return {
              ...base,
              label: labelOf(keys[0] || base.key).replace(/^画面|^图像|^摄像头/, "") || base.label,
              value: value || base.value,
              color: QUALITY_COLORS[i % QUALITY_COLORS.length],
            };
          });
          setQuality(nextQ);
        }

        const buckets = Array.from({ length: 24 }, () => 0);
        for (const a of recent) {
          const d = new Date(a.created_at);
          if (!Number.isNaN(d.getTime())) buckets[d.getHours()] += 1;
        }
        if (buckets.some((x) => x > 0)) setTrend(buckets);
      }

      const playable = taskItems
        .filter((t) => t.flv_url)
        .sort((a, b) => {
          const ar = a.runtime_status === "running" ? 0 : 1;
          const br = b.runtime_status === "running" ? 0 : 1;
          return ar - br;
        })
        .slice(0, 6);
      if (playable.length) {
        setLiveMode(true);
        setVideos(
          playable.map((t, i) => ({
            id: String(t.id),
            title: t.name || t.scene_id || `画面 ${i + 1}`,
            status: i === 0 && (alertsRecent?.meta?.total || 0) > 0 ? "alarm" : "normal",
            flvUrl: t.flv_url,
          }))
        );
      }

      // 人员态势：从事件里粗算入/出
      const enterEvents = await api
        .getEvents({ recognitionType: "01", timeFrom: from, page: 1, pageSize: 1 })
        .catch(() => null);
      const exitEvents = await api
        .getEvents({ recognitionType: "02", timeFrom: from, page: 1, pageSize: 1 })
        .catch(() => null);
      if (enterEvents?.meta?.total != null) setEnterCount(enterEvents.meta.total);
      if (exitEvents?.meta?.total != null) setExitCount(exitEvents.meta.total);

      // 地点人数：用示例结构，若有摄像头地点名则替换标签
      const sites = cameras?.items || [];
      if (sites.length) {
        const bySite = new Map<string, number>();
        for (const c of sites.slice(0, 20)) {
          const name = String(c.site_name || c.name || c.scene_id || "点位");
          bySite.set(name, (bySite.get(name) || 0) + 1);
        }
        const bars = [...bySite.entries()]
          .map(([name, value]) => ({ name, value }))
          .sort((a, b) => b.value - a.value)
          .slice(0, 4);
        if (bars.length) setPresenceBySite(bars);
      }
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 30000);
    return () => clearInterval(t);
  }, [refresh]);

  const title = useMemo(
    () => `${branding.platform_name || "DeepSight"} 实时监控大屏`,
    [branding.platform_name]
  );

  return (
    <div className="wall-screen">
      <header className="wall-head">
        <div className="wall-brand">
          <div className="wall-logo" aria-hidden />
          <div>
            <h1>{title}</h1>
            <p>{liveMode ? "已接入运行任务实时流" : "当前为示意布局，可自动填充真实数据"}</p>
          </div>
        </div>
        <div className="wall-head-right">
          <button className="wall-btn" type="button" onClick={refresh} disabled={loading}>
            {loading ? "刷新中…" : "刷新"}
          </button>
          <Link className="wall-btn" to="/live">
            实时展示
          </Link>
          <Link className="wall-btn" to="/">
            返回系统
          </Link>
          <div className="wall-clock">{formatClock(now)}</div>
          <div className="wall-status ok">系统正常</div>
        </div>
      </header>

      <div className="wall-body">
        <aside className="wall-col wall-col-left">
          <section className="wall-card">
            <h2>今日概览</h2>
            <div className="wall-kpi-grid">
              {overview.map((k) => (
                <div key={k.label} className={`wall-kpi ${k.warn ? "warn" : ""}`}>
                  <div className="wall-kpi-value">{k.value}</div>
                  <div className="wall-kpi-label">{k.label}</div>
                </div>
              ))}
            </div>
          </section>

          <section className="wall-card">
            <h2>人员态势</h2>
            <div className="wall-person-stats">
              <div>
                <div className="wall-kpi-value">{enterCount}</div>
                <div className="wall-kpi-label">入井</div>
              </div>
              <div>
                <div className="wall-kpi-value">{exitCount}</div>
                <div className="wall-kpi-label">出井</div>
              </div>
            </div>
            <BarList items={presenceBySite} color="#38bdf8" />
          </section>

          <section className="wall-card wall-card-grow">
            <h2>视频质量</h2>
            <Donut items={quality} />
          </section>
        </aside>

        <section className="wall-col wall-col-center">
          <div className="wall-video-grid">
            {videos.map((v) => (
              <article
                key={v.id}
                className={`wall-video-tile ${v.status === "alarm" ? "alarm" : ""}`}
              >
                <div className="wall-video-top">
                  <span>{v.title}</span>
                  <span className={`wall-badge ${v.status}`}>
                    {v.status === "alarm" ? "报警" : "正常"}
                  </span>
                </div>
                <div className="wall-video-body">
                  {v.flvUrl ? (
                    <StreamPlayer flvUrl={v.flvUrl} compact />
                  ) : (
                    <div className="wall-video-placeholder">
                      <div className="wall-cam-icon" />
                      <span>{v.title}</span>
                    </div>
                  )}
                </div>
              </article>
            ))}
          </div>

          <section className="wall-card wall-trend-card">
            <div className="wall-trend-head">
              <h2>24小时报警趋势</h2>
              <span className="wall-muted">示例/近时分布</span>
            </div>
            <TrendChart values={trend} />
            <div className="wall-trend-axis">
              <span>00:00</span>
              <span>06:00</span>
              <span>12:00</span>
              <span>18:00</span>
              <span>24:00</span>
            </div>
          </section>
        </section>

        <aside className="wall-col wall-col-right">
          <section className="wall-card wall-card-grow">
            <h2>最新报警</h2>
            <ul className="wall-alert-list">
              {alerts.map((a) => (
                <li key={a.id}>
                  <div className="wall-alert-thumb">
                    {a.thumb ? (
                      <img src={a.thumb} alt="" />
                    ) : (
                      <div className="wall-alert-thumb-empty" />
                    )}
                  </div>
                  <div className="wall-alert-meta">
                    <div className="wall-alert-top">
                      <span>{a.time}</span>
                      <span className="wall-level">{a.level}</span>
                    </div>
                    <div className="wall-alert-place">{a.place}</div>
                    <div className="wall-alert-type">{a.type}</div>
                  </div>
                </li>
              ))}
            </ul>
          </section>

          <section className="wall-card">
            <h2>地点报警排行</h2>
            <BarList items={siteRank} color="#22d3ee" />
          </section>

          <section className="wall-card">
            <h2>任务健康</h2>
            <div className="wall-health">
              <div className="wall-health-item ok">
                <b>{health.running}</b>
                <span>运行</span>
              </div>
              <div className="wall-health-item muted">
                <b>{health.stopped}</b>
                <span>停止</span>
              </div>
              <div className="wall-health-item warn">
                <b>{health.noVideo}</b>
                <span>无画面</span>
              </div>
            </div>
          </section>
        </aside>
      </div>
    </div>
  );
}
