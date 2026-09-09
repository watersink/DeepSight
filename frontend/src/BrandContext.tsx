import {
  ReactNode,
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";
import { api } from "./api";

export type PlatformBranding = {
  platform_name: string;
  slogan: string;
  logo_url: string | null;
};

const DEFAULT_BRANDING: PlatformBranding = {
  platform_name: "AI视频监控平台",
  slogan: "深瞳洞察每一帧 · 云端决策每一秒",
  logo_url: null,
};

type BrandingContextValue = {
  branding: PlatformBranding;
  loading: boolean;
  refresh: () => Promise<void>;
  setBranding: (b: PlatformBranding) => void;
};

const BrandingContext = createContext<BrandingContextValue | null>(null);

export function BrandingProvider({ children }: { children: ReactNode }) {
  const [branding, setBranding] = useState<PlatformBranding>(DEFAULT_BRANDING);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    try {
      const data = await api.getPlatformSettings();
      setBranding({
        platform_name: data.platform_name || DEFAULT_BRANDING.platform_name,
        slogan: data.slogan ?? DEFAULT_BRANDING.slogan,
        logo_url: data.logo_url || null,
      });
    } catch {
      /* 保持默认 */
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  useEffect(() => {
    document.title = branding.platform_name || DEFAULT_BRANDING.platform_name;
  }, [branding.platform_name]);

  const value = useMemo(
    () => ({ branding, loading, refresh, setBranding }),
    [branding, loading, refresh]
  );

  return (
    <BrandingContext.Provider value={value}>{children}</BrandingContext.Provider>
  );
}

export function useBranding() {
  const ctx = useContext(BrandingContext);
  if (!ctx) throw new Error("useBranding must be used within BrandingProvider");
  return ctx;
}
