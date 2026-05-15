import { CheckCircle2, CircleDashed, XCircle } from "lucide-react";

export function StatusPill({ status }) {
  const s = String(status || "idle");

  const ui = (() => {
    if (s === "running" || s === "queued") {
      return {
        icon: <CircleDashed size={16} />,
        text: s,
        className:
          "bg-amber-50 dark:bg-amber-900/20 text-amber-800 dark:text-amber-200 border-amber-200 dark:border-amber-900/30",
      };
    }
    if (s === "pass") {
      return {
        icon: <CheckCircle2 size={16} />,
        text: "pass",
        className:
          "bg-emerald-50 dark:bg-emerald-900/20 text-emerald-700 dark:text-emerald-200 border-emerald-200 dark:border-emerald-900/30",
      };
    }
    if (s === "fail") {
      return {
        icon: <XCircle size={16} />,
        text: "fail",
        className:
          "bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-200 border-red-200 dark:border-red-900/30",
      };
    }

    return {
      icon: null,
      text: s,
      className:
        "bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-200 border-gray-200 dark:border-gray-700",
    };
  })();

  return (
    <div
      className={`inline-flex items-center gap-2 px-2 py-1 rounded-full border text-xs font-jetbrains-mono ${ui.className}`}
    >
      {ui.icon}
      <span>{ui.text}</span>
    </div>
  );
}
