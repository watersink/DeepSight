type Props = {
  page: number;
  pageSize: number;
  total: number;
  onChange: (page: number) => void;
};

export default function Pager({ page, pageSize, total, onChange }: Props) {
  const totalPages = Math.max(1, Math.ceil(total / pageSize) || 1);
  const safePage = Math.min(Math.max(1, page), totalPages);

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
