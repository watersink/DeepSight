import { FormEvent, useEffect, useState } from "react";
import { api } from "../api";

type LlmItem = {
  id: number;
  name: string;
  enabled: boolean;
  base_url: string;
  model: string;
  temperature: number;
  max_tokens: number;
  system_prompt: string;
  has_api_key: boolean;
  api_key_masked: string;
  ready: boolean;
  configured: boolean;
};

type FormState = {
  name: string;
  base_url: string;
  model: string;
  temperature: number;
  max_tokens: number;
  system_prompt: string;
  enabled: boolean;
};

const emptyForm = (): FormState => ({
  name: "",
  base_url: "https://api.deepseek.com/v1",
  model: "deepseek-chat",
  temperature: 0.7,
  max_tokens: 2048,
  system_prompt:
    "你是 DeepSight 视频监控平台的智能助手，帮助用户理解告警、事件与系统使用。回答简洁、准确。",
  enabled: true,
});

export default function LlmSettingsPage() {
  const [items, setItems] = useState<LlmItem[]>([]);
  const [form, setForm] = useState<FormState>(emptyForm());
  const [apiKeyInput, setApiKeyInput] = useState("");
  const [editingId, setEditingId] = useState<number | null>(null);
  const [formOpen, setFormOpen] = useState(false);
  const [error, setError] = useState("");
  const [msg, setMsg] = useState("");
  const [saving, setSaving] = useState(false);

  const load = async () => {
    setError("");
    try {
      const data = await api.getLlmSettings();
      setItems(data.items || []);
    } catch (err: any) {
      setError(err.message || String(err));
    }
  };

  useEffect(() => {
    load();
  }, []);

  const openCreate = () => {
    setEditingId(null);
    setForm(emptyForm());
    setApiKeyInput("");
    setError("");
    setMsg("");
    setFormOpen(true);
  };

  const openEdit = (row: LlmItem) => {
    setEditingId(row.id);
    setForm({
      name: row.name,
      base_url: row.base_url,
      model: row.model,
      temperature: row.temperature,
      max_tokens: row.max_tokens,
      system_prompt: row.system_prompt,
      enabled: row.enabled,
    });
    setApiKeyInput("");
    setError("");
    setMsg("");
    setFormOpen(true);
  };

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setSaving(true);
    setError("");
    setMsg("");
    try {
      if (editingId == null) {
        await api.createLlmSettings({
          name: form.name.trim(),
          base_url: form.base_url.trim(),
          model: form.model.trim(),
          temperature: Number(form.temperature),
          max_tokens: Number(form.max_tokens),
          system_prompt: form.system_prompt,
          enabled: form.enabled,
          api_key: apiKeyInput.trim(),
        });
        setMsg("已新增大模型配置");
      } else {
        await api.updateLlmSettings(editingId, {
          name: form.name.trim(),
          base_url: form.base_url.trim(),
          model: form.model.trim(),
          temperature: Number(form.temperature),
          max_tokens: Number(form.max_tokens),
          system_prompt: form.system_prompt,
          enabled: form.enabled,
          api_key: apiKeyInput.trim() || undefined,
        });
        setMsg("大模型配置已更新");
      }
      setFormOpen(false);
      await load();
    } catch (err: any) {
      setError(err.message || String(err));
    } finally {
      setSaving(false);
    }
  };

  const onToggle = async (row: LlmItem) => {
    setError("");
    setMsg("");
    try {
      await api.updateLlmSettings(row.id, { enabled: !row.enabled });
      setMsg(`${row.name} 已${!row.enabled ? "启用" : "停用"}`);
      await load();
    } catch (err: any) {
      setError(err.message || String(err));
    }
  };

  const onDelete = async (row: LlmItem) => {
    if (!confirm(`确定删除「${row.name}」？`)) return;
    setError("");
    setMsg("");
    try {
      await api.deleteLlmSettings(row.id);
      setMsg("已删除");
      await load();
    } catch (err: any) {
      setError(err.message || String(err));
    }
  };

  return (
    <div className="page-shell">
      <div className="page-head">
        <div>
          <h1>大模型配置</h1>
          <p>
            可同时配置多个 OpenAI 兼容模型（DeepSeek、文心、通义等）。助手页可手动选择已启用的模型。
          </p>
        </div>
        <div className="toolbar">
          <button className="btn primary" type="button" onClick={openCreate}>
            新增模型
          </button>
          <button className="btn" type="button" onClick={() => load()}>
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
                <th>名称</th>
                <th>模型</th>
                <th>接口</th>
                <th>状态</th>
                <th>操作</th>
              </tr>
            </thead>
            <tbody>
              {items.map((row) => (
                <tr key={row.id}>
                  <td>
                    {row.name}
                    <div className="muted" style={{ fontSize: 12 }}>
                      {row.has_api_key
                        ? `Key ${row.api_key_masked}`
                        : "未配置 API Key"}
                    </div>
                  </td>
                  <td className="mono">{row.model}</td>
                  <td className="mono" style={{ maxWidth: 240, wordBreak: "break-all" }}>
                    {row.base_url}
                  </td>
                  <td>
                    <span className={`badge ${row.enabled ? "run" : "stop"}`}>
                      {row.enabled ? "启用" : "停用"}
                    </span>
                    {row.enabled && !row.ready && (
                      <div className="muted" style={{ fontSize: 12, marginTop: 4 }}>
                        缺少密钥或地址
                      </div>
                    )}
                  </td>
                  <td className="row-actions">
                    <button className="btn" type="button" onClick={() => openEdit(row)}>
                      编辑
                    </button>
                    <button
                      className={row.enabled ? "btn danger" : "btn primary"}
                      type="button"
                      onClick={() => onToggle(row)}
                    >
                      {row.enabled ? "停用" : "启用"}
                    </button>
                    <button className="btn danger" type="button" onClick={() => onDelete(row)}>
                      删除
                    </button>
                  </td>
                </tr>
              ))}
              {!items.length && (
                <tr>
                  <td colSpan={5} className="muted">
                    暂无模型配置，请点击「新增模型」
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {formOpen && (
        <div
          className="modal-backdrop"
          onClick={() => !saving && setFormOpen(false)}
          role="presentation"
        >
          <div
            className="modal-panel task-form-panel"
            onClick={(e) => e.stopPropagation()}
            role="dialog"
            aria-modal="true"
          >
            <div className="modal-head">
              <h2>{editingId == null ? "新增模型" : "编辑模型"}</h2>
              <button
                className="btn"
                type="button"
                disabled={saving}
                onClick={() => setFormOpen(false)}
              >
                关闭
              </button>
            </div>
            {error && <p className="error">{error}</p>}
            <form className="form-grid" onSubmit={onSubmit}>
              <label>
                显示名称
                <input
                  required
                  value={form.name}
                  onChange={(e) => setForm({ ...form, name: e.target.value })}
                  placeholder="如：DeepSeek / 文心"
                />
              </label>
              <label>
                状态
                <select
                  value={form.enabled ? "1" : "0"}
                  onChange={(e) =>
                    setForm({ ...form, enabled: e.target.value === "1" })
                  }
                >
                  <option value="1">启用</option>
                  <option value="0">停用</option>
                </select>
              </label>
              <label>
                模型名称
                <input
                  required
                  value={form.model}
                  onChange={(e) => setForm({ ...form, model: e.target.value })}
                  placeholder="deepseek-chat"
                />
              </label>
              <label>
                Temperature
                <input
                  type="number"
                  min={0}
                  max={2}
                  step={0.1}
                  value={form.temperature}
                  onChange={(e) =>
                    setForm({ ...form, temperature: Number(e.target.value) })
                  }
                />
              </label>
              <label className="full">
                接口地址（OpenAI 兼容，含 /v1）
                <input
                  required
                  value={form.base_url}
                  onChange={(e) => setForm({ ...form, base_url: e.target.value })}
                  placeholder="https://api.deepseek.com/v1"
                />
              </label>
              <label className="full">
                API Key
                <input
                  type="password"
                  value={apiKeyInput}
                  onChange={(e) => setApiKeyInput(e.target.value)}
                  placeholder={
                    editingId != null && items.find((i) => i.id === editingId)?.has_api_key
                      ? `已配置（留空则不修改）`
                      : "请输入 API Key"
                  }
                  autoComplete="new-password"
                  required={editingId == null}
                />
              </label>
              <label>
                Max Tokens
                <input
                  type="number"
                  min={1}
                  max={128000}
                  value={form.max_tokens}
                  onChange={(e) =>
                    setForm({ ...form, max_tokens: Number(e.target.value) })
                  }
                />
              </label>
              <label className="full">
                系统提示词
                <textarea
                  rows={4}
                  value={form.system_prompt}
                  onChange={(e) =>
                    setForm({ ...form, system_prompt: e.target.value })
                  }
                  placeholder="助手人设与回答风格"
                />
              </label>
              <div className="full" style={{ display: "flex", gap: 8 }}>
                <button className="btn primary" type="submit" disabled={saving}>
                  {saving ? "保存中…" : "保存"}
                </button>
                <button
                  className="btn"
                  type="button"
                  disabled={saving}
                  onClick={() => setFormOpen(false)}
                >
                  取消
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
