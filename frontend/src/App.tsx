import { useMemo, useState } from "react";
import { NavLink, Navigate, Route, Routes, useLocation } from "react-router-dom";
import { AuthProvider, RequireAdmin, RequireAuth, useAuth } from "./AuthContext";
import { BrandingProvider, useBranding } from "./BrandContext";
import BrandMark from "./components/BrandMark";
import CamerasPage from "./pages/CamerasPage";
import ModelsPage from "./pages/ModelsPage";
import AlgorithmsPage from "./pages/AlgorithmsPage";
import TasksPage from "./pages/TasksPage";
import LiveMonitorPage from "./pages/LiveMonitorPage";
import AlertsPage from "./pages/AlertsPage";
import EventsPage from "./pages/EventsPage";
import UsersPage from "./pages/UsersPage";
import PartnersPage from "./pages/PartnersPage";
import PlatformSettingsPage from "./pages/PlatformSettingsPage";
import LoginPage from "./pages/LoginPage";

type NavLeaf = { to: string; label: string; end?: boolean; adminOnly?: boolean };
type NavGroup = { label: string; adminOnly?: boolean; children: NavLeaf[] };
type NavEntry = NavLeaf | NavGroup;

function isGroup(item: NavEntry): item is NavGroup {
  return "children" in item;
}

const navItems: NavEntry[] = [
  { to: "/", label: "摄像头配置", end: true },
  { to: "/models", label: "模型管理" },
  { to: "/algorithms", label: "算法管理" },
  { to: "/tasks", label: "任务配置" },
  { to: "/live", label: "实时展示" },
  { to: "/events", label: "事件管理" },
  { to: "/alerts", label: "报警管理" },
  {
    label: "系统管理",
    adminOnly: true,
    children: [
      { to: "/system/basic", label: "基础配置", adminOnly: true },
      { to: "/system/partners", label: "第三方接入配置", adminOnly: true },
      { to: "/users", label: "用户管理", adminOnly: true },
    ],
  },
];

function AppShell() {
  const { user, logout, isAdmin } = useAuth();
  const { branding } = useBranding();
  const location = useLocation();
  const systemOpenDefault = location.pathname.startsWith("/system") || location.pathname === "/users";
  const [systemOpen, setSystemOpen] = useState(systemOpenDefault);

  const visibleNav = useMemo(
    () =>
      navItems
        .filter((item) => !item.adminOnly || isAdmin)
        .map((item) => {
          if (!isGroup(item)) return item;
          return {
            ...item,
            children: item.children.filter((c) => !c.adminOnly || isAdmin),
          };
        })
        .filter((item) => !isGroup(item) || item.children.length > 0),
    [isAdmin]
  );

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <BrandMark logoUrl={branding.logo_url} />
          <div>
            <div className="brand-title">{branding.platform_name}</div>
            <div className="brand-sub">{branding.slogan}</div>
          </div>
        </div>
        <nav>
          {visibleNav.map((item) => {
            if (!isGroup(item)) {
              return (
                <NavLink
                  key={item.to}
                  to={item.to}
                  end={item.end}
                  className={({ isActive }) => (isActive ? "nav-item active" : "nav-item")}
                >
                  {item.label}
                </NavLink>
              );
            }
            const childActive = item.children.some(
              (c) =>
                location.pathname === c.to ||
                (c.to !== "/" && location.pathname.startsWith(c.to))
            );
            const open = systemOpen || childActive;
            return (
              <div key={item.label} className="nav-group">
                <button
                  type="button"
                  className={`nav-group-label ${childActive ? "active" : ""}`}
                  onClick={() => setSystemOpen((v) => !v)}
                >
                  <span>{item.label}</span>
                  <span className="nav-group-caret">{open ? "-" : "+"}</span>
                </button>
                {open &&
                  item.children.map((c) => (
                    <NavLink
                      key={c.to}
                      to={c.to}
                      className={({ isActive }) =>
                        isActive ? "nav-item nav-sub active" : "nav-item nav-sub"
                      }
                    >
                      {c.label}
                    </NavLink>
                  ))}
              </div>
            );
          })}
        </nav>
        <div className="sidebar-user">
          <div className="sidebar-user-name">
            {user?.display_name || user?.username}
            <span className="muted">
              {" "}
              · {user?.role === "admin" ? "管理员" : "用户"}
            </span>
          </div>
          <button className="btn" type="button" onClick={logout}>
            退出登录
          </button>
        </div>
        <a className="docs-link" href="/docs" target="_blank" rel="noreferrer">
          API 文档
        </a>
      </aside>
      <main className="main">
        <Routes>
          <Route path="/" element={<CamerasPage />} />
          <Route path="/models" element={<ModelsPage />} />
          <Route path="/algorithms" element={<AlgorithmsPage />} />
          <Route path="/tasks" element={<TasksPage />} />
          <Route path="/live" element={<LiveMonitorPage />} />
          <Route path="/events" element={<EventsPage />} />
          <Route path="/alerts" element={<AlertsPage />} />
          <Route
            path="/system/basic"
            element={
              <RequireAdmin>
                <PlatformSettingsPage />
              </RequireAdmin>
            }
          />
          <Route
            path="/system/partners"
            element={
              <RequireAdmin>
                <PartnersPage />
              </RequireAdmin>
            }
          />
          <Route
            path="/users"
            element={
              <RequireAdmin>
                <UsersPage />
              </RequireAdmin>
            }
          />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    </div>
  );
}

export default function App() {
  return (
    <AuthProvider>
      <BrandingProvider>
        <Routes>
          <Route path="/login" element={<LoginPage />} />
          <Route
            path="/*"
            element={
              <RequireAuth>
                <AppShell />
              </RequireAuth>
            }
          />
        </Routes>
      </BrandingProvider>
    </AuthProvider>
  );
}
