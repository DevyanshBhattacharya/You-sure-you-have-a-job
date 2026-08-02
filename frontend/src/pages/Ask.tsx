import { useEffect, useRef, useState } from "react";
import { streamChat } from "../api/client";
import type { Citation } from "../api/types";
import { Card } from "../components/ui";

interface Turn {
  role: "user" | "model";
  content: string;
  citations?: Citation[];
  tools?: string[];
}

const SUGGESTIONS = [
  "Which companies have rejected me?",
  "How many applications are in the interview stage?",
  "What do I need to do this week?",
  "What did the recruiter ask me to send?",
];

/**
 * Seconds since `active` became true.
 *
 * A local model can take minutes to answer, during which the stream is silent.
 * Without a moving number the page is indistinguishable from a hung request.
 */
function useElapsed(active: boolean): number {
  const [seconds, setSeconds] = useState(0);

  useEffect(() => {
    if (!active) return;
    setSeconds(0);
    const started = Date.now();
    const id = setInterval(() => setSeconds(Math.floor((Date.now() - started) / 1000)), 1000);
    return () => clearInterval(id);
  }, [active]);

  return seconds;
}

function Citations({ citations }: { citations: Citation[] }) {
  if (citations.length === 0) return null;
  return (
    <div className="mt-3 border-t border-slate-200 pt-2 dark:border-slate-800">
      <p className="mb-1.5 text-[11px] font-semibold tracking-wide text-slate-400 uppercase dark:text-slate-500">
        Sources
      </p>
      <ul className="space-y-1">
        {citations.map((c, i) => (
          <li key={i} className="text-xs text-slate-500 dark:text-slate-400">
            {c.company && <span className="font-medium">{c.company}</span>}
            {c.company && c.subject && " — "}
            {c.subject}
            {c.received_at && (
              <span className="ml-1 text-slate-400 dark:text-slate-600">
                {new Date(c.received_at).toLocaleDateString()}
              </span>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}

export default function Ask() {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [activeTool, setActiveTool] = useState<string | null>(null);

  const abortRef = useRef<AbortController | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const elapsed = useElapsed(streaming);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [turns, activeTool]);

  useEffect(() => () => abortRef.current?.abort(), []);

  async function send(question: string) {
    const trimmed = question.trim();
    if (!trimmed || streaming) return;

    // History excludes the turn being asked; the backend appends it.
    const history = turns.map((t) => ({ role: t.role, content: t.content }));

    setTurns((prev) => [...prev, { role: "user", content: trimmed }, { role: "model", content: "" }]);
    setInput("");
    setStreaming(true);
    setActiveTool(null);

    const controller = new AbortController();
    abortRef.current = controller;

    const appendToLast = (patch: (turn: Turn) => Turn) =>
      setTurns((prev) => {
        const next = [...prev];
        next[next.length - 1] = patch(next[next.length - 1]);
        return next;
      });

    try {
      await streamChat(
        trimmed,
        history,
        (event) => {
          switch (event.type) {
            case "tool":
              setActiveTool(event.name);
              break;
            case "token":
              appendToLast((turn) => ({ ...turn, content: turn.content + event.text }));
              break;
            case "done":
              appendToLast((turn) => ({
                ...turn,
                citations: event.citations,
                tools: event.tool_calls,
              }));
              break;
            case "error":
              appendToLast((turn) => ({
                ...turn,
                content: turn.content + (turn.content ? "\n\n" : "") + event.message,
              }));
              break;
          }
        },
        controller.signal,
      );
    } catch (err) {
      if ((err as Error).name !== "AbortError") {
        appendToLast((turn) => ({
          ...turn,
          content: turn.content || `Request failed: ${(err as Error).message}`,
        }));
      }
    } finally {
      setStreaming(false);
      setActiveTool(null);
      abortRef.current = null;
    }
  }

  return (
    <div className="mx-auto flex h-[calc(100vh-11rem)] max-w-3xl flex-col">
      <div className="flex-1 space-y-4 overflow-y-auto pr-1">
        {turns.length === 0 && (
          <Card className="p-6">
            <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
              Ask about your job search
            </h2>
            <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
              Answers come from your own tracked mail — both the text of messages and the
              structured application data.
            </p>
            <div className="mt-4 flex flex-wrap gap-2">
              {SUGGESTIONS.map((s) => (
                <button
                  key={s}
                  type="button"
                  onClick={() => void send(s)}
                  className="rounded-full border border-slate-300 px-3 py-1.5 text-xs text-slate-600 transition hover:border-slate-400 hover:bg-slate-50 dark:border-slate-700 dark:text-slate-400 dark:hover:bg-slate-800"
                >
                  {s}
                </button>
              ))}
            </div>
          </Card>
        )}

        {turns.map((turn, i) =>
          turn.role === "user" ? (
            <div key={i} className="flex justify-end">
              <div className="max-w-[85%] rounded-2xl rounded-br-sm bg-slate-900 px-4 py-2.5 text-sm text-white dark:bg-slate-100 dark:text-slate-900">
                {turn.content}
              </div>
            </div>
          ) : (
            <div key={i} className="flex justify-start">
              <Card className="max-w-[90%] px-4 py-3">
                <div className="text-sm leading-relaxed whitespace-pre-wrap text-slate-800 dark:text-slate-200">
                  {turn.content}
                  {streaming && i === turns.length - 1 && !turn.content && (
                    <span className="text-slate-400 italic">
                      Thinking… {elapsed > 2 && `${elapsed}s`}
                    </span>
                  )}
                </div>
                {turn.citations && <Citations citations={turn.citations} />}
              </Card>
            </div>
          ),
        )}

        {activeTool && (
          <p className="text-xs text-slate-400 dark:text-slate-500">
            <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-sky-500 align-middle" />
            <span className="ml-2 font-mono">{activeTool}</span> …
            {elapsed > 2 && <span className="ml-1">{elapsed}s</span>}
          </p>
        )}

        <div ref={bottomRef} />
      </div>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          void send(input);
        }}
        className="mt-4 flex gap-2"
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about applications, interviews, deadlines…"
          disabled={streaming}
          className="flex-1 rounded-lg border border-slate-300 bg-white px-4 py-2.5 text-sm text-slate-900 placeholder:text-slate-400 focus:ring-2 focus:ring-sky-500 focus:outline-none disabled:opacity-60 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100"
        />
        {streaming ? (
          <button
            type="button"
            onClick={() => abortRef.current?.abort()}
            className="rounded-lg border border-slate-300 px-4 py-2.5 text-sm font-medium text-slate-600 hover:bg-slate-100 dark:border-slate-700 dark:text-slate-400 dark:hover:bg-slate-800"
          >
            Stop
          </button>
        ) : (
          <button
            type="submit"
            disabled={!input.trim()}
            className="rounded-lg bg-slate-900 px-4 py-2.5 text-sm font-medium text-white transition hover:bg-slate-800 disabled:opacity-40 dark:bg-slate-100 dark:text-slate-900 dark:hover:bg-white"
          >
            Ask
          </button>
        )}
      </form>
    </div>
  );
}
