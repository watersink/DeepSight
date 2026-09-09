import { CSSProperties, FormEvent, useMemo, useState } from "react";
import { Navigate, useLocation, useNavigate } from "react-router-dom";
import { useAuth } from "../AuthContext";
import { useBranding } from "../BrandContext";
import BrandMark from "../components/BrandMark";

/** 简单可复现伪随机，保证刷新后装饰位形稳定 */
function mulberry32(seed: number) {
  let t = seed >>> 0;
  return () => {
    t += 0x6d2b79f5;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

function buildField(seed = 20260907) {
  const rnd = mulberry32(seed);
  const dots = Array.from({ length: 64 }, (_, i) => ({
    id: i,
    x: rnd() * 100,
    y: rnd() * 100,
    r: 0.6 + rnd() * 2.2,
    opacity: 0.18 + rnd() * 0.55,
    delay: rnd() * 5,
    dur: 2.8 + rnd() * 4.2,
  }));
  const lines = Array.from({ length: 28 }, (_, i) => {
    const x1 = rnd() * 100;
    const y1 = rnd() * 100;
    const len = 4 + rnd() * 14;
    const ang = rnd() * Math.PI * 2;
    return {
      id: i,
      x1,
      y1,
      x2: x1 + Math.cos(ang) * len,
      y2: y1 + Math.sin(ang) * len,
      opacity: 0.1 + rnd() * 0.28,
      delay: rnd() * 6,
      dur: 4 + rnd() * 5,
    };
  });
  const blobs = Array.from({ length: 7 }, (_, i) => ({
    id: i,
    x: rnd() * 100,
    y: rnd() * 100,
    s: 80 + rnd() * 180,
    opacity: 0.04 + rnd() * 0.08,
  }));
  return { dots, lines, blobs };
}

export default function LoginPage() {
  const { user, loading, login } = useAuth();
  const { branding } = useBranding();
  const navigate = useNavigate();
  const location = useLocation();
  const from = (location.state as any)?.from || "/";
  const field = useMemo(() => buildField(), []);

  const [username, setUsername] = useState("admin");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);

  if (!loading && user) {
    return <Navigate to={from} replace />;
  }

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError("");
    setSubmitting(true);
    try {
      await login(username.trim(), password);
      navigate(from, { replace: true });
    } catch (err: any) {
      setError(err.message || String(err));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="login-page">
      <div className="login-bg" aria-hidden="true">
        <div className="login-bg-gradient" />
        <svg className="login-bg-field" viewBox="0 0 100 100" preserveAspectRatio="xMidYMid slice">
          {field.blobs.map((b) => (
            <circle
              key={`b-${b.id}`}
              className="login-blob"
              cx={b.x}
              cy={b.y}
              r={b.s / 10}
              style={{ opacity: b.opacity }}
            />
          ))}
          {field.lines.map((l) => (
            <line
              key={`l-${l.id}`}
              className="login-seg"
              x1={l.x1}
              y1={l.y1}
              x2={l.x2}
              y2={l.y2}
              style={
                {
                  opacity: l.opacity,
                  "--d": `${l.delay}s`,
                  "--dur": `${l.dur}s`,
                } as CSSProperties
              }
            />
          ))}
          {field.dots.map((d) => (
            <circle
              key={`d-${d.id}`}
              className="login-dot"
              cx={d.x}
              cy={d.y}
              r={0.12 + d.r * 0.14}
              style={
                {
                  opacity: d.opacity,
                  "--d": `${d.delay}s`,
                  "--dur": `${d.dur}s`,
                } as CSSProperties
              }
            />
          ))}
        </svg>
        <div className="login-bg-scan" />
        <div className="login-bg-noise" />
      </div>

      <div className="login-card">
        <div className="login-card-edge" aria-hidden="true" />
        <div className="login-brand">
          <BrandMark className="login-mark" logoUrl={branding.logo_url} />
          <div>
            <h1>{branding.platform_name}</h1>
            <p className="muted">{branding.slogan}</p>
          </div>
        </div>

        <div className="login-status mono muted" aria-hidden="true">
          <span className="login-status-dot" />
          SYS READY · SECURE CHANNEL
        </div>

        <form onSubmit={onSubmit} className="login-form">
          <label>
            用户名
            <input
              autoFocus
              autoComplete="username"
              required
              value={username}
              onChange={(e) => setUsername(e.target.value)}
            />
          </label>
          <label>
            密码
            <input
              type="password"
              autoComplete="current-password"
              required
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
          </label>
          {error && <p className="error">{error}</p>}
          <button className="btn primary" type="submit" disabled={submitting}>
            {submitting ? "认证中…" : "登录系统"}
          </button>
        </form>

        <p className="muted login-hint">
          默认管理员：admin / Admin@123（首次启动自动创建）
        </p>
      </div>
    </div>
  );
}
