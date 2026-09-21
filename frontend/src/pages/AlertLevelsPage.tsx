import { useEffect, useState } from "react";
import { api } from "../api";
import AlertLevelBadge, {
  ALERT_LEVELS,
  type AlertLevel,
} from "../components/AlertLevelBadge";

type AlertLevelRow = {
  type_key: string;
  name_zh: string;
  category: string;
  level: number;
  sort_order: number;
};

const LEVEL_OPTIONS: { level: AlertLevel; label: string }[] = ([1, 2, 3] as AlertLevel[]).map(
  (level) => ({
    level,
    label: `${ALERT_LEVELS[level].rank}${ALERT_LEVELS[level].name}`,
  })
);

export default function AlertLevelsPage() {
  const [items, setItems] = useState<AlertLevelRow[]>([]);
  const [error, setError] = useState("");
  const [msg, setMsg] = useState("");
  const [savingKey, setSavingKey] = useState("");

  const load = async () => {
    setError("");
    try {
      setItems(await api.getAlertLevels());
    } catch (err: any) {
      setError(err.message || String(err));
    }
  };

  useEffect(() => {
    load();
  }, []);

  const onChangeLevel = async (row: AlertLevelRow, level: number) => {
    if (level === row.level) return;
    setSavingKey(row.type_key);
    setError("");
    setMsg("");
    try {
      const updated = await api.updateAlertLevel(row.type_key, level);
      setItems((current) =>
        current.map((item) => (item.type_key === updated.type_key ? updated : item))
      );
      setMsg(`${row.name_zh} 已改为 ${ALERT_LEVELS[level as AlertLevel].rank}${ALERT_LEVELS[level as AlertLevel].name}，仅影响之后的新记录`);
    } catch (err: any) {
      setError(err.message || String(err));
    } finally {
      setSavingKey("");
    }
  };

  return (
    <div className="page-shell">
      <div className="page-head">
        <div>
          <h1>报警等级管理</h1>
          <p>
            按识别类型配置严重程度。进事件管理还是报警管理不随等级改变。修改只影响之后新产生的记录。
          </p>
        </div>
      </div>

      {error && <p className="error">{error}</p>}
      {msg && <p className="ok">{msg}</p>}

      <div className="card page-list-card">
        <div className="page-list-scroll">
          <table>
            <thead>
              <tr>
                <th>类型键</th>
                <th>中文</th>
                <th>列表</th>
                <th>等级</th>
              </tr>
            </thead>
            <tbody>
              {items.map((row) => (
                <tr key={row.type_key}>
                  <td className="mono">{row.type_key}</td>
                  <td>{row.name_zh}</td>
                  <td>{row.category === "event" ? "事件" : "报警"}</td>
                  <td>
                    <select
                      value={row.level}
                      disabled={savingKey === row.type_key}
                      onChange={(e) => onChangeLevel(row, Number(e.target.value))}
                    >
                      {LEVEL_OPTIONS.map((option) => (
                        <option key={option.level} value={option.level}>
                          {option.label}
                        </option>
                      ))}
                    </select>
                    <div style={{ marginTop: 6 }}>
                      <AlertLevelBadge
                        level={
                          row.level === 1 || row.level === 2 || row.level === 3
                            ? row.level
                            : 3
                        }
                      />
                    </div>
                  </td>
                </tr>
              ))}
              {!items.length && (
                <tr>
                  <td colSpan={4} className="muted">
                    暂无类型
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
