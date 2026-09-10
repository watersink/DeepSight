import { clearAuth, getToken } from "./auth";

export type PageMeta = { total: number; page: number; page_size: number };

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const token = getToken();
  const headers: Record<string, string> = {
    ...(init?.headers as Record<string, string> | undefined),
  };
  // FormData 由浏览器自动带 multipart boundary，勿强制 JSON
  const isForm = typeof FormData !== "undefined" && init?.body instanceof FormData;
  if (!isForm && !headers["Content-Type"]) {
    headers["Content-Type"] = "application/json";
  }
  if (token) headers.Authorization = `Bearer ${token}`;

  const res = await fetch(path, {
    ...init,
    headers,
  });

  if (res.status === 401) {
    const isLogin = path.includes("/auth/login");
    if (!isLogin) {
      clearAuth();
      if (!window.location.pathname.startsWith("/login")) {
        window.location.assign("/login");
      }
    }
  }

  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || JSON.stringify(body);
    } catch {
      /* ignore */
    }
    throw new ApiError(
      res.status,
      typeof detail === "string" ? detail : JSON.stringify(detail)
    );
  }
  if (res.status === 204) return undefined as T;
  return res.json();
}

export const api = {
  login: (username: string, password: string) =>
    request<{
      access_token: string;
      token_type: string;
      expires_in: number;
      user: any;
    }>("/api/v1/auth/login", {
      method: "POST",
      body: JSON.stringify({ username, password }),
    }),
  me: () => request<any>("/api/v1/auth/me"),
  changePassword: (old_password: string, new_password: string) =>
    request("/api/v1/auth/change-password", {
      method: "POST",
      body: JSON.stringify({ old_password, new_password }),
    }),
  getUsers: (opts?: { page?: number; pageSize?: number }) => {
    const q = new URLSearchParams();
    q.set("page", String(opts?.page || 1));
    q.set("page_size", String(opts?.pageSize || 10));
    return request<{ items: any[]; meta: PageMeta }>(`/api/v1/users?${q}`);
  },
  createUser: (body: object) =>
    request("/api/v1/users", { method: "POST", body: JSON.stringify(body) }),
  updateUser: (id: number, body: object) =>
    request(`/api/v1/users/${id}`, {
      method: "PATCH",
      body: JSON.stringify(body),
    }),
  deleteUser: (id: number) =>
    request(`/api/v1/users/${id}`, { method: "DELETE" }),

  getCameras: () => request<{ items: any[]; meta: PageMeta }>("/api/v1/cameras"),
  getCameraTree: () =>
    request<{ mines: any[]; unassigned: any[] }>("/api/v1/camera-tree"),
  createCamera: (body: object) =>
    request("/api/v1/cameras", { method: "POST", body: JSON.stringify(body) }),
  updateCamera: (id: number, body: object) =>
    request(`/api/v1/cameras/${id}`, { method: "PATCH", body: JSON.stringify(body) }),
  deleteCamera: (id: number) =>
    request(`/api/v1/cameras/${id}`, { method: "DELETE" }),

  getMines: () => request<{ items: any[] }>("/api/v1/mines"),
  createMine: (body: object) =>
    request("/api/v1/mines", { method: "POST", body: JSON.stringify(body) }),
  updateMine: (id: number, body: object) =>
    request(`/api/v1/mines/${id}`, { method: "PATCH", body: JSON.stringify(body) }),
  deleteMine: (id: number) =>
    request(`/api/v1/mines/${id}`, { method: "DELETE" }),

  getSites: (mineId?: number) => {
    const q = mineId != null ? `?mine_id=${mineId}` : "";
    return request<{ items: any[] }>(`/api/v1/sites${q}`);
  },
  createSite: (body: object) =>
    request("/api/v1/sites", { method: "POST", body: JSON.stringify(body) }),
  updateSite: (id: number, body: object) =>
    request(`/api/v1/sites/${id}`, { method: "PATCH", body: JSON.stringify(body) }),
  deleteSite: (id: number) =>
    request(`/api/v1/sites/${id}`, { method: "DELETE" }),

  getAlgorithms: () =>
    request<{ items: any[]; meta: PageMeta }>("/api/v1/algorithm-configs"),
  createAlgorithm: (body: object) =>
    request("/api/v1/algorithm-configs", { method: "POST", body: JSON.stringify(body) }),
  updateAlgorithm: (id: number, body: object) =>
    request(`/api/v1/algorithm-configs/${id}`, {
      method: "PATCH",
      body: JSON.stringify(body),
    }),
  deleteAlgorithm: (id: number) =>
    request(`/api/v1/algorithm-configs/${id}`, { method: "DELETE" }),

  getSkills: () => request<{ skills: any[] }>("/api/v1/skills"),

  getModels: () =>
    request<{
      server_url: string;
      server_live: boolean;
      server_ready: boolean;
      server_name?: string | null;
      server_version?: string | null;
      items: any[];
      workers?: any[];
      error?: string | null;
    }>("/api/v1/models"),

  getTasks: (opts?: { page?: number; pageSize?: number }) => {
    const q = new URLSearchParams();
    q.set("page", String(opts?.page || 1));
    q.set("page_size", String(opts?.pageSize || 10));
    return request<{ items: any[]; meta: PageMeta }>(`/api/v1/task-configs?${q}`);
  },
  createTask: (body: object) =>
    request("/api/v1/task-configs", { method: "POST", body: JSON.stringify(body) }),
  updateTask: (id: number, body: object) =>
    request(`/api/v1/task-configs/${id}`, { method: "PATCH", body: JSON.stringify(body) }),
  deleteTask: (id: number) =>
    request(`/api/v1/task-configs/${id}`, { method: "DELETE" }),
  startTask: (id: number) =>
    request(`/api/v1/task-configs/${id}/start`, { method: "POST" }),
  stopTask: (id: number) =>
    request(`/api/v1/task-configs/${id}/stop`, { method: "POST" }),

  getWorkers: () => request<{ items: any[]; total: number }>("/api/v1/workers"),
  getWorkerSkills: (workerId: string) =>
    request<{ worker_id: string; skills: any[] }>(
      `/api/v1/workers/${encodeURIComponent(workerId)}/skills`
    ),
  getWorkerModels: (workerId: string) =>
    request<any>(`/api/v1/workers/${encodeURIComponent(workerId)}/models`),
  getAlerts: (opts?: {
    status?: string;
    sceneId?: string;
    skillName?: string;
    recognitionType?: string;
    timeFrom?: string;
    timeTo?: string;
    page?: number;
    pageSize?: number;
  }) => {
    const q = new URLSearchParams();
    q.set("category", "alert");
    if (opts?.status) q.set("status", opts.status);
    if (opts?.sceneId) q.set("scene_id", opts.sceneId);
    if (opts?.skillName) q.set("skill_name", opts.skillName);
    if (opts?.recognitionType) q.set("recognition_type", opts.recognitionType);
    if (opts?.timeFrom) q.set("time_from", opts.timeFrom);
    if (opts?.timeTo) q.set("time_to", opts.timeTo);
    q.set("page", String(opts?.page || 1));
    q.set("page_size", String(opts?.pageSize || 10));
    return request<{ items: any[]; meta: PageMeta }>(`/api/v1/alerts?${q}`);
  },
  updateAlert: (id: number, status: string) =>
    request(`/api/v1/alerts/${id}`, {
      method: "PATCH",
      body: JSON.stringify({ status }),
    }),

  getEvents: (opts?: {
    status?: string;
    sceneId?: string;
    skillName?: string;
    recognitionType?: string;
    timeFrom?: string;
    timeTo?: string;
    page?: number;
    pageSize?: number;
  }) => {
    const q = new URLSearchParams();
    if (opts?.status) q.set("status", opts.status);
    if (opts?.sceneId) q.set("scene_id", opts.sceneId);
    if (opts?.skillName) q.set("skill_name", opts.skillName);
    if (opts?.recognitionType) q.set("recognition_type", opts.recognitionType);
    if (opts?.timeFrom) q.set("time_from", opts.timeFrom);
    if (opts?.timeTo) q.set("time_to", opts.timeTo);
    q.set("page", String(opts?.page || 1));
    q.set("page_size", String(opts?.pageSize || 10));
    return request<{ items: any[]; meta: PageMeta }>(`/api/v1/events?${q}`);
  },
  updateEvent: (id: number, status: string) =>
    request(`/api/v1/events/${id}`, {
      method: "PATCH",
      body: JSON.stringify({ status }),
    }),

  getPartners: (opts?: { page?: number; pageSize?: number }) => {
    const q = new URLSearchParams();
    q.set("page", String(opts?.page || 1));
    q.set("page_size", String(opts?.pageSize || 10));
    return request<{ items: any[]; meta: PageMeta }>(`/api/v1/partners?${q}`);
  },
  createPartner: (body: object) =>
    request("/api/v1/partners", { method: "POST", body: JSON.stringify(body) }),
  updatePartner: (id: number, body: object) =>
    request(`/api/v1/partners/${id}`, {
      method: "PATCH",
      body: JSON.stringify(body),
    }),
  deletePartner: (id: number) =>
    request(`/api/v1/partners/${id}`, { method: "DELETE" }),
  testPartnerWebhook: (id: number) =>
    request<{ ok: boolean; payload: any }>(`/api/v1/partners/${id}/test-webhook`, {
      method: "POST",
    }),

  getPlatformSettings: () =>
    request<{
      platform_name: string;
      slogan: string;
      logo_url?: string | null;
    }>("/api/v1/platform-settings"),
  updatePlatformSettings: (body: {
    platform_name?: string;
    slogan?: string;
    clear_logo?: boolean;
  }) =>
    request<{
      platform_name: string;
      slogan: string;
      logo_url?: string | null;
    }>("/api/v1/platform-settings", {
      method: "PATCH",
      body: JSON.stringify(body),
    }),
  uploadPlatformLogo: (file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    return request<{
      platform_name: string;
      slogan: string;
      logo_url?: string | null;
    }>("/api/v1/platform-settings/logo", {
      method: "POST",
      body: fd,
    });
  },
};
