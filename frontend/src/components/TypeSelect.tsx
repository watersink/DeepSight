import { useEffect, useRef, useState } from "react";

export type TypeOption = {
  value: string;
  code?: string;
  desc?: string;
  label?: string;
};

export function TypeLines({
  codes,
  labels,
  fallback = "-",
}: {
  codes: string[];
  labels: Record<string, string>;
  fallback?: string;
}) {
  if (!codes.length) return <>{fallback}</>;
  return (
    <>
      {codes.map((code) => (
        <div key={code}>
          <div className="mono">{code}</div>
          {labels[code] ? <div className="muted">{labels[code]}</div> : null}
        </div>
      ))}
    </>
  );
}

export default function TypeSelect({
  value,
  onChange,
  options,
}: {
  value: string;
  onChange: (value: string) => void;
  options: TypeOption[];
}) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);
  const selected = options.find((item) => item.value === value) || options[0];

  useEffect(() => {
    const onDoc = (event: MouseEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, []);

  return (
    <div className="type-filter" ref={rootRef}>
      <button
        type="button"
        className="type-filter-btn"
        aria-expanded={open}
        onClick={() => setOpen((current) => !current)}
      >
        <TypeOptionView item={selected} />
      </button>
      {open && (
        <div className="type-filter-menu" role="listbox">
          {options.map((item) => (
            <button
              key={item.value || "all"}
              type="button"
              role="option"
              aria-selected={item.value === value}
              className={item.value === value ? "active" : ""}
              onClick={() => {
                onChange(item.value);
                setOpen(false);
              }}
            >
              <TypeOptionView item={item} />
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

function TypeOptionView({ item }: { item?: TypeOption }) {
  if (item?.code) {
    return (
      <>
        <span className="mono">{item.code}</span>
        {item.desc ? <span className="muted">{item.desc}</span> : null}
      </>
    );
  }
  return <span>{item?.label || "全部类型"}</span>;
}
