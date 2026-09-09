import { FormEvent, useEffect, useState } from "react";
import { api } from "../api";
import { useBranding } from "../BrandContext";
import BrandMark from "../components/BrandMark";

export default function PlatformSettingsPage() {
  const { branding, setBranding, refresh } = useBranding();
  const [name, setName] = useState(branding.platform_name);
  const [slogan, setSlogan] = useState(branding.slogan);
  const [logoUrl, setLogoUrl] = useState<string | null>(branding.logo_url);
  const [error, setError] = useState("");
  const [msg, setMsg] = useState("");
  const [saving, setSaving] = useState(false);
  const [uploading, setUploading] = useState(false);

  useEffect(() => {
    setName(branding.platform_name);
    setSlogan(branding.slogan);
    setLogoUrl(branding.logo_url);
  }, [branding]);

  const onSave = async (e: FormEvent) => {
    e.preventDefault();
    setSaving(true);
    setError("");
    setMsg("");
    try {
      const data = await api.updatePlatformSettings({
        platform_name: name.trim(),
        slogan: slogan.trim(),
      });
      const next = {
        platform_name: data.platform_name,
        slogan: data.slogan,
        logo_url: data.logo_url || null,
      };
      setBranding(next);
      setMsg("基础配置已保存");
    } catch (err: any) {
      setError(err.message || String(err));
    } finally {
      setSaving(false);
    }
  };

  const onUpload = async (file: File | null) => {
    if (!file) return;
    setUploading(true);
    setError("");
    setMsg("");
    try {
      const data = await api.uploadPlatformLogo(file);
      const next = {
        platform_name: data.platform_name,
        slogan: data.slogan,
        logo_url: data.logo_url || null,
      };
      setLogoUrl(next.logo_url);
      setBranding(next);
      setMsg("Logo 已更新");
    } catch (err: any) {
      setError(err.message || String(err));
    } finally {
      setUploading(false);
    }
  };

  const onClearLogo = async () => {
    if (!confirm("恢复为默认图标？")) return;
    setSaving(true);
    setError("");
    try {
      const data = await api.updatePlatformSettings({ clear_logo: true });
      const next = {
        platform_name: data.platform_name,
        slogan: data.slogan,
        logo_url: null,
      };
      setLogoUrl(null);
      setBranding(next);
      setMsg("已恢复默认图标");
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
          <h1>基础配置</h1>
          <p>自定义平台名称与图标，登录页与侧栏品牌区将同步展示。</p>
        </div>
        <div className="toolbar">
          <button className="btn" type="button" onClick={() => refresh()}>
            刷新
          </button>
        </div>
      </div>

      {error && <p className="error">{error}</p>}
      {msg && <p className="ok">{msg}</p>}

      <div className="card page-list-card">
        <div className="page-list-scroll" style={{ padding: 4 }}>
          <h3 className="section-title" style={{ marginTop: 0 }}>
            预览
          </h3>
          <div className="brand" style={{ padding: "8px 0 16px" }}>
            <BrandMark logoUrl={logoUrl} />
            <div>
              <div className="brand-title">{name || "平台名称"}</div>
              <div className="brand-sub">{slogan || "口号"}</div>
            </div>
          </div>

          <form className="form-grid" onSubmit={onSave}>
            <label className="full">
              平台名称
              <input
                required
                maxLength={128}
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="AI视频监控平台"
              />
            </label>
            <label className="full">
              平台口号
              <input
                maxLength={255}
                value={slogan}
                onChange={(e) => setSlogan(e.target.value)}
                placeholder="深瞳洞察每一帧 · 云端决策每一秒"
              />
            </label>
            <label className="full">
              平台图标
              <input
                type="file"
                accept="image/png,image/jpeg,image/webp,image/gif,image/svg+xml"
                disabled={uploading}
                onChange={(e) => onUpload(e.target.files?.[0] || null)}
              />
              <span className="muted" style={{ marginTop: 6, fontSize: 12 }}>
                支持 png / jpg / webp / gif / svg，不超过 2MB
              </span>
            </label>
            <div className="full row-actions">
              <button className="btn primary" type="submit" disabled={saving || uploading}>
                {saving ? "保存中…" : "保存名称与口号"}
              </button>
              <button
                className="btn"
                type="button"
                disabled={saving || uploading || !logoUrl}
                onClick={onClearLogo}
              >
                恢复默认图标
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}
