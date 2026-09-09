/** AI视频监控平台品牌图标：自定义 Logo 或默认镜头智瞳 */

export default function BrandMark({
  className = "",
  logoUrl,
}: {
  className?: string;
  logoUrl?: string | null;
}) {
  if (logoUrl) {
    return (
      <div className={`brand-mark brand-mark-image ${className}`.trim()} aria-hidden="true">
        <img src={logoUrl} alt="" className="brand-mark-img" />
      </div>
    );
  }

  return (
    <div className={`brand-mark ${className}`.trim()} aria-hidden="true">
      <svg
        className="brand-mark-icon"
        viewBox="0 0 40 40"
        width="22"
        height="22"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
      >
        <rect x="5" y="9" width="30" height="22" rx="4" stroke="currentColor" strokeWidth="2" />
        <circle cx="20" cy="20" r="7.5" stroke="currentColor" strokeWidth="2" />
        <circle cx="20" cy="20" r="3" fill="currentColor" />
        <path
          d="M12 20h4M24 20h4"
          stroke="currentColor"
          strokeWidth="1.6"
          strokeLinecap="round"
          opacity="0.85"
        />
        <circle cx="29" cy="13.5" r="1.4" fill="currentColor" />
      </svg>
    </div>
  );
}
