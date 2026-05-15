import { Clock } from "lucide-react";
import { eventBadge, humanMeetingType } from "@/utils/calendarHelpers";

export function UpcomingEventCard({ event, onClick }) {
  const badge = eventBadge(event);
  const label = humanMeetingType(event);

  return (
    <button
      type="button"
      onClick={() => onClick(event)}
      className="w-full text-left rounded-2xl border border-black/10 dark:border-white/10 bg-white/70 dark:bg-white/5 backdrop-blur p-4 hover:bg-white/80 dark:hover:bg-white/10 transition-colors"
    >
      <div className="flex items-start justify-between mb-2">
        <h4 className="font-semibold text-gray-900 dark:text-gray-50 font-jetbrains-mono">
          {label}
        </h4>
        <span
          className={`px-2 py-1 text-xs rounded-full font-medium ${badge.className}`}
        >
          {badge.text}
        </span>
      </div>
      <div className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
        <div className="flex items-center font-jetbrains-mono">
          <Clock size={14} className="mr-2 text-[var(--brand)]" />
          {new Date(event.starts_at).toLocaleString("en-US")}
        </div>
        {event.client_name ? (
          <div className="text-xs font-jetbrains-mono text-gray-600 dark:text-gray-300">
            {event.client_name}
          </div>
        ) : null}
      </div>
    </button>
  );
}
