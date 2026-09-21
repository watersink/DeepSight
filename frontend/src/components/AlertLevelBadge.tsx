/** 三级告警：一级最严重，三级最轻。 */
export const ALERT_LEVELS = {
  1: { rank: "一级", name: "严重", className: "level-1" },
  2: { rank: "二级", name: "警告", className: "level-2" },
  3: { rank: "三级", name: "提示", className: "level-3" },
} as const;

export type AlertLevel = keyof typeof ALERT_LEVELS;

export function normalizeAlertLevel(value: unknown, fallback: AlertLevel): AlertLevel {
  const level = Number(value);
  if (level === 1 || level === 2 || level === 3) return level;
  return fallback;
}

export default function AlertLevelBadge({ level }: { level: AlertLevel }) {
  const item = ALERT_LEVELS[level];
  return (
    <span className={`badge ${item.className}`}>
      {item.rank} {item.name}
    </span>
  );
}
