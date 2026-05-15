import { Clock } from "lucide-react";
import { eventBadge, humanMeetingType } from "@/utils/calendarHelpers";

export function EventCard({ event, onClick }) {
  const starts = new Date(event.starts_at);
  const when = starts.toLocaleString("en-US");
  const badge = eventBadge(event);
  const label = humanMeetingType(event);

  return (
    <button
      type="button"
      onClick={() => onClick(event)}
      className="w-full text-left rounded-2xl border border-black/10 dark:border-white/10 bg-white/70 dark:bg-white/5 backdrop-blur p-4 hover:bg-white/80 dark:hover:bg-white/10 transition-colors"
    >
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0">
          <div className="font-semibold text-gray-900 dark:text-gray-50 font-jetbrains-mono truncate">
            {label}
          </div>
          <div className="mt-1 flex items-center text-sm text-gray-700 dark:text-gray-300 font-jetbrains-mono">
            <Clock size={14} className="mr-2 text-[var(--brand)]" />
            {when}
          </div>
          {event.client_name ? (
            <div className="mt-1 text-xs text-gray-600 dark:text-gray-300 font-jetbrains-mono">
              Client: {event.client_name}
            </div>
          ) : null}
          {event.property_title ? (
            <div className="mt-1 text-xs text-gray-600 dark:text-gray-300 font-jetbrains-mono">
              Property: {event.property_title}
            </div>
          ) : null}
        </div>
        <span
          className={`px-2 py-1 text-xs rounded-full font-medium ${badge.className}`}
        >
          {badge.text}
        </span>
      </div>
    </button>
  );
}
