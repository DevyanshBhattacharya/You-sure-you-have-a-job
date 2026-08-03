import { useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useRef, useState } from "react";
import { getToken } from "../api/client";
import type { Notification, SocketMessage } from "../api/types";

export type ConnectionState = "connecting" | "open" | "closed";

const MAX_TOASTS = 4;
const TOAST_MS = 9000;
const BACKOFF_START_MS = 1000;
const BACKOFF_MAX_MS = 30_000;

export interface Toast {
  id: number;
  title: string;
  body: string | null;
  priority: Notification["priority"];
}

/**
 * Holds the dashboard WebSocket.
 *
 * On an inbound event it invalidates the relevant queries rather than merging
 * a partial payload into the cache — the socket says *what changed*, and the
 * REST endpoints remain the single source of truth for *the current value*.
 */
export function useNotifications() {
  const queryClient = useQueryClient();
  const [state, setState] = useState<ConnectionState>("connecting");
  const [toasts, setToasts] = useState<Toast[]>([]);

  const socketRef = useRef<WebSocket | null>(null);
  const retryRef = useRef(BACKOFF_START_MS);
  const timerRef = useRef<number | null>(null);
  const closedByUs = useRef(false);

  const dismissToast = useCallback((id: number) => {
    setToasts((current) => current.filter((t) => t.id !== id));
  }, []);

  const pushToast = useCallback((notification: Notification) => {
    setToasts((current) => [
      ...current.slice(-(MAX_TOASTS - 1)),
      {
        id: notification.id,
        title: notification.title,
        body: notification.body,
        priority: notification.priority,
      },
    ]);
    window.setTimeout(() => {
      setToasts((current) => current.filter((t) => t.id !== notification.id));
    }, TOAST_MS);
  }, []);

  const handle = useCallback(
    (message: SocketMessage) => {
      switch (message.topic) {
        case "notification.created": {
          const notification = message.data as Notification;
          pushToast(notification);
          queryClient.invalidateQueries({ queryKey: ["notifications"] });
          queryClient.invalidateQueries({ queryKey: ["board"] });
          queryClient.invalidateQueries({ queryKey: ["stats"] });
          break;
        }
        case "application.updated":
          queryClient.invalidateQueries({ queryKey: ["board"] });
          queryClient.invalidateQueries({ queryKey: ["stats"] });
          queryClient.invalidateQueries({ queryKey: ["applications"] });
          break;
        case "email.processed":
          queryClient.invalidateQueries({ queryKey: ["emails"] });
          queryClient.invalidateQueries({ queryKey: ["health"] });
          break;
        default:
          break;
      }
    },
    [pushToast, queryClient],
  );

  useEffect(() => {
    closedByUs.current = false;

    const connect = () => {
      const scheme = window.location.protocol === "https:" ? "wss" : "ws";
      // Query string, not a header: the browser WebSocket API gives no way to
      // set headers on the handshake. The server also checks Origin, since
      // CORS does not apply to WebSockets at all.
      const token = getToken();
      const auth = token ? `?token=${encodeURIComponent(token)}` : "";
      const socket = new WebSocket(`${scheme}://${window.location.host}/ws/notifications${auth}`);
      socketRef.current = socket;
      setState("connecting");

      socket.onopen = () => {
        setState("open");
        retryRef.current = BACKOFF_START_MS;
      };

      socket.onmessage = (event) => {
        try {
          handle(JSON.parse(event.data) as SocketMessage);
        } catch {
          /* ignore malformed frame */
        }
      };

      socket.onerror = () => socket.close();

      socket.onclose = () => {
        setState("closed");
        if (closedByUs.current) return;

        // Exponential backoff with jitter, so a backend restart doesn't get
        // hammered by every open tab reconnecting in lockstep.
        const delay = Math.min(retryRef.current, BACKOFF_MAX_MS);
        retryRef.current = Math.min(delay * 2, BACKOFF_MAX_MS);
        timerRef.current = window.setTimeout(connect, delay + Math.random() * 500);
      };
    };

    connect();

    return () => {
      closedByUs.current = true;
      if (timerRef.current !== null) window.clearTimeout(timerRef.current);
      socketRef.current?.close();
    };
  }, [handle]);

  return { state, toasts, dismissToast };
}
