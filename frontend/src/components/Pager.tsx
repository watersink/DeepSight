import { FormEvent, useState } from "react";

type Props = {
  page: number;
  pageSize: number;
  total: number;
  onChange: (page: number) => void;
};

export default function Pager({ page, pageSize, total, onChange }: Props) {
  const totalPages = Math.max(1, Math.ceil(total / pageSize) || 1);
  const safePage = Math.min(Math.max(1, page), totalPages);
  const [target, setTarget] = useState("");

  const jump = () => {
    const raw = target.trim();
    if (!raw) return;
    const n = Number(raw);
    if (!Number.isFinite(n)) {
      setTarget("");
      return;
    }
    const next = Math.min(totalPages, Math.max(1, Math.trunc(n)));
    setTarget("");
    if (next !== safePage) onChange(next);
  };

  const onSubmit = (e: FormEvent) => {
    e.preventDefault();
    jump();
  };

  return (
    <div className="pager">
      <button
        className="btn"
        type="button"
        disabled={safePage <= 1}
        onClick={() => onChange(safePage - 1)}
      >
        上一页
      </button>
      <span className="muted">
        第 {safePage} / {totalPages} 页
        {total ? ` · 共 ${total} 条 · 每页 ${pageSize}` : ""}
      </span>
      <form className="pager-jump" onSubmit={onSubmit}>
        <input
          className="pager-jump-input"
          type="number"
          min={1}
          max={totalPages}
          inputMode="numeric"
          placeholder="页码"
          aria-label="输入要访问的页码"
          value={target}
          onChange={(e) => setTarget(e.target.value)}
        />
        <button className="btn" type="submit">
          跳转
        </button>
      </form>
      <button
        className="btn"
        type="button"
        disabled={safePage >= totalPages}
        onClick={() => onChange(safePage + 1)}
      >
        下一页
      </button>
    </div>
  );
}
