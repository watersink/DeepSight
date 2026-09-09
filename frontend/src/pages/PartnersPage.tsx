import { FormEvent, useEffect, useState } from "react";
import { api } from "../api";
import Pager from "../components/Pager";

const PAGE_SIZE = 10;

const TYPE_OPTIONS = [
  { value: "01", label: "01 人员计数（入）" },
  { value: "02", label: "02 人员计数（出）" },
  { value: "04", label: "04 非常规通道入井" },
  { value: "05", label: "05 非常规通道出井" },
  { value: "06", label: "06 入井闸机出闸" },
  { value: "07", label: "07 出井闸机入闸" },
];

const emptyForm = {
  name: "",
  code: "",
  webhook_url: "",
  enabled: true,
  subscribe_alert: true,
  subscribe_event: false,
  recognition_types: [] as string[],
  scene_ids: "",
  remark: "",
  rotate_api_key: false,
  rotate_webhook_secret: false,
};

function joinTypes(types: string[] | null | undefined) {
  return (types || []).filter(Boolean).join(",") || "不限";
}

export default function PartnersPage() {
  const [items, setItems] = useState<any[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [error, setError] = useState("");
  const [msg, setMsg] = useState("");
  const [formOpen, setFormOpen] = useState(false);
  const [editingId, setEditingId] = useState<number | null>(null);
  const [form, setForm] = useState(emptyForm);
  const [saving, setSaving] = useState(false);

  const [editing, setEditing] = useState<any | null>(null);

  const load = async (pageNo = page) => {
    setError("");
    try {
      const data = await api.getPartners({ page: pageNo, pageSize: PAGE_SIZE });
      setItems(data.items || []);
      setTotal(data.meta?.total ?? 0);
    } catch (e: any) {
      setError(e.message || String(e));
    }
  };

  useEffect(() => {
    load(page);
  }, [page]);

  const openCreate = () => {
    setEditingId(null);
    setEditing(null);
    setForm(emptyForm);
    setFormOpen(true);
    setError("");
    setMsg("");
  };

  const openEdit = (p: any) => {
    const cats: string[] = p.subscribe_categories || ["alert"];
    setEditingId(p.id);
    setEditing(p);
    setForm({
      name: p.name || "",
      code: p.code || "",
      webhook_url: p.webhook_url || "",
      enabled: !!p.enabled,
      subscribe_alert: cats.includes("alert"),
      subscribe_event: cats.includes("event"),
      recognition_types: Array.isArray(p.subscribe_recognition_types)
        ? [...p.subscribe_recognition_types]
        : [],
      scene_ids: Array.isArray(p.subscribe_scene_ids)
        ? p.subscribe_scene_ids.join(",")
        : "",
      remark: p.remark || "",
      rotate_api_key: false,
      rotate_webhook_secret: false,
    });
    setFormOpen(true);
    setError("");
    setMsg("");
  };

  const buildBody = () => {
    const categories: string[] = [];
    if (form.subscribe_alert) categories.push("alert");
    if (form.subscribe_event) categories.push("event");
    const scenes = form.scene_ids
      .split(/[,，\s]+/)
      .map((s) => s.trim())
      .filter(Boolean);
    return {
      name: form.name.trim(),
      code: form.code.trim(),
      webhook_url: form.webhook_url.trim(),
      enabled: form.enabled,
      subscribe_categories: categories.length ? categories : ["alert"],
      subscribe_recognition_types: form.recognition_types,
      subscribe_scene_ids: scenes,
      remark: form.remark,
    };
  };

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setSaving(true);
    setError("");
    setMsg("");
    try {
      if (editingId) {
        const body: any = buildBody();
        delete body.code;
        body.rotate_api_key = form.rotate_api_key;
        body.rotate_webhook_secret = form.rotate_webhook_secret;
        await api.updatePartner(editingId, body);
        setMsg("接入配置已更新");
      } else {
        if (!form.code.trim()) throw new Error("请填写接入编码");
        await api.createPartner(buildBody());
        setMsg("接入配置已创建（请保存 API Key 与签名密钥并交给对方）");
        setPage(1);
      }
      setFormOpen(false);
      await load(editingId ? page : 1);
    } catch (err: any) {
      setError(err.message || String(err));
    } finally {
      setSaving(false);
    }
  };

  const toggleType = (code: string) => {
    setForm((f) => {
      const has = f.recognition_types.includes(code);
      return {
        ...f,
        recognition_types: has
          ? f.recognition_types.filter((t) => t !== code)
          : [...f.recognition_types, code],
      };
    });
  };

  return (
    <div className="page-shell">
      <div className="page-head">
        <div>
          <h1>第三方接入配置</h1>
          <p>
            配置 Webhook 回调地址与订阅范围。对方用 API Key 拉取详情/媒体；实时通知由本系统按订阅过滤后推送。
          </p>
        </div>
        <div className="toolbar">
          <button className="btn primary" type="button" onClick={openCreate}>
            新增接入
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
                <th>名称</th>
                <th>编码</th>
                <th>Webhook</th>
                <th>订阅</th>
                <th>状态</th>
                <th>操作</th>
              </tr>
            </thead>
            <tbody>
              {items.map((p) => (
                <tr key={p.id}>
                  <td>
                    {p.name}
                    <div className="muted mono" style={{ fontSize: 11 }}>
                      key: {p.api_key}
                    </div>
                  </td>
                  <td className="mono">{p.code}</td>
                  <td className="mono" style={{ maxWidth: 220, wordBreak: "break-all" }}>
                    {p.webhook_url || <span className="muted">未配置</span>}
                  </td>
                  <td>
                    <div className="muted">
                      {(p.subscribe_categories || ["alert"]).join(",")}
                    </div>
                    <div className="mono" style={{ fontSize: 12 }}>
                      类型 {joinTypes(p.subscribe_recognition_types)}
                    </div>
                  </td>
                  <td>
                    <span className={`badge ${p.enabled ? "run" : "stop"}`}>
                      {p.enabled ? "启用" : "禁用"}
                    </span>
                  </td>
                  <td className="row-actions">
                    <button className="btn" type="button" onClick={() => openEdit(p)}>
                      编辑
                    </button>
                    <button
                      className="btn"
                      type="button"
                      disabled={!p.webhook_url}
                      onClick={async () => {
                        try {
                          await api.testPartnerWebhook(p.id);
                          setMsg(`已向 ${p.name} 发送测试 Webhook`);
                          setError("");
                        } catch (err: any) {
                          setError(err.message || String(err));
                        }
                      }}
                    >
                      测试推送
                    </button>
                    <button
                      className="btn danger"
                      type="button"
                      onClick={async () => {
                        if (!confirm(`删除接入 ${p.name}?`)) return;
                        try {
                          await api.deletePartner(p.id);
                          await load(page);
                          setMsg("已删除");
                        } catch (err: any) {
                          setError(err.message || String(err));
                        }
                      }}
                    >
                      删除
                    </button>
                  </td>
                </tr>
              ))}
              {!items.length && (
                <tr>
                  <td colSpan={6} className="muted">
                    暂无接入配置。新增后即可热生效，无需重启服务。
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
        <Pager page={page} pageSize={PAGE_SIZE} total={total} onChange={setPage} />
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
              <h2>{editingId ? "编辑接入" : "新增接入"}</h2>
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
            {editingId && editing && (
              <div className="card" style={{ marginBottom: 12, padding: 12 }}>
                <p className="muted" style={{ marginBottom: 6 }}>
                  交给对方的凭证（请妥善保管）
                </p>
                <p className="mono" style={{ fontSize: 12, wordBreak: "break-all" }}>
                  API Key: {editing.api_key}
                </p>
                <p className="mono" style={{ fontSize: 12, wordBreak: "break-all" }}>
                  Webhook Secret: {editing.webhook_secret}
                </p>
                <p className="muted" style={{ marginTop: 6, fontSize: 12 }}>
                  拉取详情：Header <code>X-API-Key</code> →{" "}
                  <code>/open/v1/alerts/{"{alert_id}"}</code>
                </p>
              </div>
            )}
            <form className="form-grid" onSubmit={onSubmit}>
              <label>
                名称
                <input
                  required
                  value={form.name}
                  onChange={(e) => setForm({ ...form, name: e.target.value })}
                  placeholder="如：某某平台"
                />
              </label>
              <label>
                接入编码
                <input
                  required
                  disabled={!!editingId}
                  value={form.code}
                  onChange={(e) => setForm({ ...form, code: e.target.value })}
                  placeholder="字母数字_-，创建后不可改"
                />
              </label>
              <label className="full">
                Webhook URL
                <input
                  value={form.webhook_url}
                  onChange={(e) => setForm({ ...form, webhook_url: e.target.value })}
                  placeholder="https://partner.example.com/api/alerts/webhook"
                />
              </label>
              <label>
                订阅类别
                <div className="toolbar" style={{ marginTop: 6 }}>
                  <label style={{ flexDirection: "row", alignItems: "center", gap: 6 }}>
                    <input
                      type="checkbox"
                      checked={form.subscribe_alert}
                      onChange={(e) =>
                        setForm({ ...form, subscribe_alert: e.target.checked })
                      }
                    />
                    报警 alert
                  </label>
                  <label style={{ flexDirection: "row", alignItems: "center", gap: 6 }}>
                    <input
                      type="checkbox"
                      checked={form.subscribe_event}
                      onChange={(e) =>
                        setForm({ ...form, subscribe_event: e.target.checked })
                      }
                    />
                    事件 event
                  </label>
                </div>
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
              <label className="full">
                订阅识别类型（不选表示不限）
                <div className="toolbar" style={{ marginTop: 6 }}>
                  {TYPE_OPTIONS.map((t) => (
                    <label
                      key={t.value}
                      style={{ flexDirection: "row", alignItems: "center", gap: 4 }}
                    >
                      <input
                        type="checkbox"
                        checked={form.recognition_types.includes(t.value)}
                        onChange={() => toggleType(t.value)}
                      />
                      {t.label}
                    </label>
                  ))}
                </div>
              </label>
              <label className="full">
                订阅场景 ID（逗号分隔，空=不限）
                <input
                  value={form.scene_ids}
                  onChange={(e) => setForm({ ...form, scene_ids: e.target.value })}
                  placeholder="scene_001,scene_002"
                />
              </label>
              <label className="full">
                备注
                <input
                  value={form.remark}
                  onChange={(e) => setForm({ ...form, remark: e.target.value })}
                />
              </label>
              {editingId && (
                <>
                  <label style={{ flexDirection: "row", alignItems: "center", gap: 8 }}>
                    <input
                      type="checkbox"
                      checked={form.rotate_api_key}
                      onChange={(e) =>
                        setForm({ ...form, rotate_api_key: e.target.checked })
                      }
                    />
                    重新生成 API Key
                  </label>
                  <label style={{ flexDirection: "row", alignItems: "center", gap: 8 }}>
                    <input
                      type="checkbox"
                      checked={form.rotate_webhook_secret}
                      onChange={(e) =>
                        setForm({ ...form, rotate_webhook_secret: e.target.checked })
                      }
                    />
                    重新生成 Webhook 签名密钥
                  </label>
                </>
              )}
              <div className="full row-actions" style={{ marginTop: 8 }}>
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
