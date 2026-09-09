const TOKEN_KEY = "yj_access_token";
const USER_KEY = "yj_auth_user";

export type AuthUser = {
  id: number;
  username: string;
  display_name: string;
  role: string;
  enabled: boolean;
  remark?: string;
};

export function getToken(): string | null {
  try {
    return localStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}

export function setAuth(token: string, user: AuthUser) {
  localStorage.setItem(TOKEN_KEY, token);
  localStorage.setItem(USER_KEY, JSON.stringify(user));
}

export function clearAuth() {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(USER_KEY);
}

export function getStoredUser(): AuthUser | null {
  try {
    const raw = localStorage.getItem(USER_KEY);
    if (!raw) return null;
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

export function isAdmin(user?: AuthUser | null): boolean {
  return (user || getStoredUser())?.role === "admin";
}
