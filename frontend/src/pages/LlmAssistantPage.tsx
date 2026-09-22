import { FormEvent, useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api";
import { getToken } from "../auth";
import { useAuth } from "../AuthContext";

type ChatRole = "user" | "assistant";
type ChatMessage = { id: string; role: ChatRole; content: string };
type LlmOption = { id: number; name: string; model: string; base_url: string };

const SELECTED_LLM_KEY = "deepsight_llm_selected_id";

function uid() {
  return `${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
}

async function streamChat(
  llmId: number,
  messages: { role: string; content: string }[],
  onDelta: (text: string) => void,
  signal?: AbortSignal
) {
  const token = getToken();
  const res = await fetch("/api/v1/llm/chat/sse", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: JSON.stringify({ messages, llm_id: llmId }),
    signal,
  });

  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || JSON.stringify(body);
    } catch {
      /* ignore */
    }
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  if (!res.body) throw new Error("浏览器不支持流式响应");

  const reader = res.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  let eventName = "message";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const chunks = buffer.split("\n");
    buffer = chunks.pop() || "";
    for (const raw of chunks) {
      const line = raw.replace(/\r$/, "");
      if (!line) {
        eventName = "message";
        continue;
      }
      if (line.startsWith("event:")) {
        eventName = line.slice(6).trim();
        continue;
      }
      if (!line.startsWith("data:")) continue;
      const payload = line.slice(5).trim();
      if (!payload) continue;
      let data: any = {};
      try {
        data = JSON.parse(payload);
      } catch {
        continue;
      }
      if (eventName === "delta" && data.content) {
        onDelta(String(data.content));
      } else if (eventName === "error") {
        throw new Error(data.message || "对话失败");
      }
    }
  }
}

export default function LlmAssistantPage() {
  const { isAdmin } = useAuth();
  const [options, setOptions] = useState<LlmOption[]>([]);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [loadingOptions, setLoadingOptions] = useState(true);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const listRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  const loadOptions = async () => {
    setLoadingOptions(true);
    setError("");
    try {
      const data = await api.getLlmOptions();
      const items = data.items || [];
      setOptions(items);
      const saved = Number(localStorage.getItem(SELECTED_LLM_KEY) || 0);
      const preferred =
        items.find((item) => item.id === saved) || items[0] || null;
      setSelectedId(preferred ? preferred.id : null);
    } catch (err: any) {
      setError(err.message || String(err));
      setOptions([]);
      setSelectedId(null);
    } finally {
      setLoadingOptions(false);
    }
  };

  useEffect(() => {
    loadOptions();
    return () => abortRef.current?.abort();
  }, []);

  useEffect(() => {
    const el = listRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, busy]);

  const selected = options.find((item) => item.id === selectedId) || null;
  const canChat = !!selected;
  const hasConversation = messages.length > 0;

  const onSelectModel = (id: number) => {
    setSelectedId(id);
    localStorage.setItem(SELECTED_LLM_KEY, String(id));
  };

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault();
    const text = input.trim();
    if (!text || busy || !selected) return;

    setError("");
    setInput("");
    const userMsg: ChatMessage = { id: uid(), role: "user", content: text };
    const assistantId = uid();
    setMessages((current) => [
      ...current,
      userMsg,
      { id: assistantId, role: "assistant", content: "" },
    ]);
    setBusy(true);

    const history = [...messages, userMsg].map((m) => ({
      role: m.role,
      content: m.content,
    }));
    const controller = new AbortController();
    abortRef.current = controller;

    try {
      await streamChat(
        selected.id,
        history,
        (delta) => {
          setMessages((current) =>
            current.map((m) =>
              m.id === assistantId ? { ...m, content: m.content + delta } : m
            )
          );
        },
        controller.signal
      );
      setMessages((current) =>
        current.map((m) =>
          m.id === assistantId && !m.content
            ? { ...m, content: "（模型未返回内容）" }
            : m
        )
      );
    } catch (err: any) {
      if (err?.name === "AbortError") return;
      const message = err.message || String(err);
      setError(message);
      setMessages((current) =>
        current.map((m) =>
          m.id === assistantId
            ? { ...m, content: m.content || `出错了：${message}` }
            : m
        )
      );
    } finally {
      setBusy(false);
      abortRef.current = null;
    }
  };

  const onNewChat = () => {
    abortRef.current?.abort();
    setMessages([]);
    setError("");
    setBusy(false);
  };

  return (
    <div className="llm-assist-page">
      <div className="llm-assist-shell">
        <header className="llm-assist-head">
          <div>
            <div className="llm-assist-title">大模型助手</div>
            <div className="llm-assist-sub">
              {selected
                ? `当前：${selected.name}（${selected.model}）`
                : "对话式分析告警与系统问题"}
            </div>
          </div>
          <div className="llm-assist-actions">
            <select
              className="llm-assist-model-select"
              value={selectedId ?? ""}
              disabled={busy || !options.length}
              onChange={(e) => onSelectModel(Number(e.target.value))}
              title="选择模型"
            >
              {!options.length && <option value="">暂无可用模型</option>}
              {options.map((item) => (
                <option key={item.id} value={item.id}>
                  {item.name} · {item.model}
                </option>
              ))}
            </select>
            {hasConversation && (
              <button className="btn" type="button" onClick={onNewChat} disabled={busy}>
                新对话
              </button>
            )}
          </div>
        </header>

        {!canChat && (
          <div className="llm-assist-empty">
            <h2>尚未就绪</h2>
            <p>
              {loadingOptions
                ? "正在加载可用模型…"
                : "请先在系统管理中新增并启用至少一个大模型配置。"}
            </p>
            {isAdmin && !loadingOptions && (
              <Link className="btn primary" to="/system/llm">
                去配置
              </Link>
            )}
          </div>
        )}

        {canChat && !hasConversation && (
          <div className="llm-assist-empty">
            <h2>有什么可以帮你？</h2>
            <p>可以问告警含义、过线规则，或平台使用方式。右上角可切换已启用的模型。</p>
            <div className="llm-assist-hints">
              {[
                "非常规通道入井是什么意思？",
                "事件管理和报警管理有什么区别？",
                "画面人数统计是怎么触发的？",
              ].map((hint) => (
                <button
                  key={hint}
                  type="button"
                  className="llm-assist-hint"
                  onClick={() => setInput(hint)}
                >
                  {hint}
                </button>
              ))}
            </div>
          </div>
        )}

        {hasConversation && (
          <div className="llm-assist-messages" ref={listRef}>
            <div className="llm-assist-messages-inner">
              {messages.map((m) => (
                <div
                  key={m.id}
                  className={`llm-assist-bubble ${m.role}${
                    busy && m.role === "assistant" && !m.content ? " typing" : ""
                  }`}
                >
                  <div className="llm-assist-bubble-role">
                    {m.role === "user" ? "我" : "助手"}
                  </div>
                  <div className="llm-assist-bubble-body">
                    {m.content || (busy ? "思考中…" : "")}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {error && <p className="error llm-assist-error">{error}</p>}

        <form className="llm-assist-composer" onSubmit={onSubmit}>
          <div className="llm-assist-composer-inner">
            <textarea
              rows={3}
              value={input}
              disabled={!canChat || busy}
              placeholder={
                canChat ? "输入问题，Enter 发送，Shift+Enter 换行" : "请先完成并启用大模型配置"
              }
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  if (!busy && input.trim()) {
                    (e.currentTarget.form as HTMLFormElement | null)?.requestSubmit();
                  }
                }
              }}
            />
            <button
              className="btn primary"
              type="submit"
              disabled={!canChat || busy || !input.trim()}
            >
              {busy ? "生成中…" : "发送"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
