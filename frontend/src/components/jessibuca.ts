/** Jessibuca 加载与 FLV 地址工具（多路复用） */

export type JessibucaPlayer = {
  play: (url: string) => void;
  pause?: () => void;
  destroy: () => void;
  on: (event: string, callback: (value?: any) => void) => void;
  setVolume?: (v: number) => void;
  mute?: () => void;
};

export type JessibucaConstructor = new (options: Record<string, any>) => JessibucaPlayer;

declare global {
  interface Window {
    Jessibuca?: JessibucaConstructor;
    jessibuca?: JessibucaConstructor;
  }
}

const JESSIBUCA_SCRIPT = "/jessibuca/jessibuca.js";

export function resolveJessibuca(): JessibucaConstructor | undefined {
  return window.Jessibuca || window.jessibuca;
}

export function decoderUrl(): string {
  return `${window.location.origin}/jessibuca/decoder.js`;
}

/** Jessibuca 对 WS-FLV 更稳；ZLM 同一路径同时支持 http/ws */
export function toPlayUrl(httpFlv: string): string {
  try {
    const u = new URL(httpFlv);
    if (u.protocol === "http:") u.protocol = "ws:";
    else if (u.protocol === "https:") u.protocol = "wss:";
    return u.toString();
  } catch {
    return httpFlv;
  }
}

export function loadJessibucaScript(): Promise<JessibucaConstructor> {
  const existingCtor = resolveJessibuca();
  if (existingCtor) {
    return Promise.resolve(existingCtor);
  }
  return new Promise((resolve, reject) => {
    const existing = document.querySelector<HTMLScriptElement>(
      'script[data-jessibuca="1"]'
    );
    if (existing) {
      existing.addEventListener("load", () => {
        const ctor = resolveJessibuca();
        if (ctor) resolve(ctor);
        else reject(new Error("Jessibuca 加载后不可用"));
      });
      existing.addEventListener("error", () =>
        reject(new Error("Jessibuca 脚本加载失败"))
      );
      return;
    }
    const script = document.createElement("script");
    script.src = JESSIBUCA_SCRIPT;
    script.async = true;
    script.dataset.jessibuca = "1";
    script.onload = () => {
      const ctor = resolveJessibuca();
      if (ctor) resolve(ctor);
      else reject(new Error("Jessibuca 加载后不可用"));
    };
    script.onerror = () =>
      reject(new Error("Jessibuca 脚本加载失败，请确认 public/jessibuca 资源存在"));
    document.head.appendChild(script);
  });
}
