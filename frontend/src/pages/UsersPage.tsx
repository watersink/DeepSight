import { FormEvent, useEffect, useState } from "react";
import { api } from "../api";
import { useAuth } from "../AuthContext";
import Pager from "../components/Pager";

const PAGE_SIZE = 10;

const emptyForm = {
  username: "",
  display_name: "",
  password: "",
  role: "user",
  enabled: true,
  remark: "",
};

export default function UsersPage() {
  const { user: me } = useAuth();
  const [items, setItems] = useState<any[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [error, setError] = useState("");
  const [msg, setMsg] = useState("");
  const [formOpen, setFormOpen] = useState(false);
  const [editingId, setEditingId] = useState<number | null>(null);
  const [form, setForm] = useState(emptyForm);
  const [saving, setSaving] = useState(false);

  const load = async (pageNo = page) => {
    setError("");
    try {
      const data = await api.getUsers({ page: pageNo, pageSize: PAGE_SIZE });
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
    setForm(emptyForm);
    setFormOpen(true);
    setError("");
    setMsg("");
  };

  const openEdit = (u: any) => {
    setEditingId(u.id);
    setForm({
      username: u.username,
      display_name: u.display_name || "",
      password: "",
      role: u.role || "user",
      enabled: !!u.enabled,
      remark: u.remark || "",
    });
    setFormOpen(true);
    setError("");
    setMsg("");
  };

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setSaving(true);
    setError("");
    setMsg("");
    try {
      if (editingId) {
        const body: any = {
          display_name: form.display_name,
          role: form.role,
          enabled: form.enabled,
          remark: form.remark,
        };
        if (form.password.trim()) body.password = form.password;
        await api.updateUser(editingId, body);
        setMsg("用户已更新");
      } else {
        if (!form.password.trim()) throw new Error("请设置初始密码");
        await api.createUser({
          username: form.username.trim(),
          password: form.password,
          display_name: form.display_name || form.username,
          role: form.role,
          enabled: form.enabled,
          remark: form.remark,
        });
        setMsg("用户已创建");
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

  return (
    <div className="page-shell">
      <div className="page-head">
        <div>
          <h1>用户管理</h1>
          <p>管理系统登录账号。仅管理员可访问本页。</p>
        </div>
        <div className="toolbar">
          <button className="btn primary" type="button" onClick={openCreate}>
            新增用户
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
                <th>用户名</th>
                <th>显示名</th>
                <th>角色</th>
                <th>状态</th>
                <th>最近登录</th>
                <th>操作</th>
              </tr>
            </thead>
            <tbody>
              {items.map((u) => (
                <tr key={u.id}>
                  <td>{u.id}</td>
                  <td className="mono">{u.username}</td>
                  <td>{u.display_name || "—"}</td>
                  <td>
                    <span className={`badge ${u.role === "admin" ? "run" : "stop"}`}>
                      {u.role === "admin" ? "管理员" : "普通用户"}
                    </span>
                  </td>
                  <td>
                    <span className={`badge ${u.enabled ? "run" : "stop"}`}>
                      {u.enabled ? "启用" : "禁用"}
                    </span>
                  </td>
                  <td className="mono muted">{u.last_login_at || "—"}</td>
                  <td className="row-actions">
                    <button className="btn" type="button" onClick={() => openEdit(u)}>
                      编辑
                    </button>
                    <button
                      className="btn danger"
                      type="button"
                      disabled={u.id === me?.id}
                      onClick={async () => {
                        if (!confirm(`删除用户 ${u.username}?`)) return;
                        try {
                          await api.deleteUser(u.id);
                          await load(page);
                          setMsg("用户已删除");
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
                  <td colSpan={7} className="muted">
                    暂无用户
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
              <div>
                <h2>{editingId ? "编辑用户" : "新增用户"}</h2>
              </div>
              <button
                className="btn"
                type="button"
                disabled={saving}
                onClick={() => setFormOpen(false)}
              >
                关闭
              </button>
            </div>
            <div className="modal-body">
              {error && <p className="error">{error}</p>}
              <form onSubmit={onSubmit}>
                <div className="form-grid">
                  <label>
                    用户名
                    <input
                      required
                      disabled={!!editingId}
                      value={form.username}
                      onChange={(e) => setForm({ ...form, username: e.target.value })}
                    />
                  </label>
                  <label>
                    显示名
                    <input
                      value={form.display_name}
                      onChange={(e) =>
                        setForm({ ...form, display_name: e.target.value })
                      }
                    />
                  </label>
                  <label>
                    {editingId ? "新密码（留空不改）" : "初始密码"}
                    <input
                      type="password"
                      required={!editingId}
                      minLength={editingId ? undefined : 6}
                      value={form.password}
                      onChange={(e) => setForm({ ...form, password: e.target.value })}
                    />
                  </label>
                  <label>
                    角色
                    <select
                      value={form.role}
                      onChange={(e) => setForm({ ...form, role: e.target.value })}
                    >
                      <option value="admin">管理员</option>
                      <option value="user">普通用户</option>
                    </select>
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
                      <option value="0">禁用</option>
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
                <div className="toolbar" style={{ marginTop: 16 }}>
                  <button className="btn primary" type="submit" disabled={saving}>
                    {saving ? "保存中…" : "保存"}
                  </button>
                  <button
                    className="btn ghost"
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
        </div>
      )}
    </div>
  );
}
