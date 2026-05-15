import {
  Clock,
  Pencil,
  User as UserIcon,
  Phone,
  Mail,
  Home,
} from "lucide-react";
import { ModalShell } from "./ModalShell";
import { eventBadge, humanMeetingType } from "@/utils/calendarHelpers";

export function EventDetailModal({ open, event, onClose, onEdit }) {
  if (!open || !event) return null;

  const when = new Date(event.starts_at).toLocaleString("en-US");
  const label = humanMeetingType(event);
  const badge = eventBadge(event);

  const durationMin = Math.max(
    0,
    Math.round((new Date(event.ends_at) - new Date(event.starts_at)) / 60000),
  );
  const durationLabel = durationMin ? `${durationMin} min` : null;

  const clientEmail = event.client_email ? String(event.client_email) : null;
  const clientPhone = event.client_phone ? String(event.client_phone) : null;

  const boxClass =
    "rounded-lg border border-black/10 dark:border-white/10 bg-white/70 dark:bg-white/5 p-3";

  return (
    <ModalShell title="Meeting details" onClose={onClose}>
      <div>
        <div className="flex flex-wrap items-center gap-2">
          <span
            className={`px-2 py-1 text-xs rounded-full font-medium ${badge.className}`}
          >
            {badge.text}
          </span>
          <div className="text-sm font-semibold text-gray-900 dark:text-gray-50 font-jetbrains-mono">
            {label}
          </div>
        </div>

        <div className="mt-4 grid grid-cols-1 sm:grid-cols-2 gap-3">
          <div className={boxClass}>
            <div className="text-xs text-gray-600 dark:text-gray-400 font-jetbrains-mono">
              Time
            </div>
            <div className="mt-1 flex items-center text-sm text-gray-900 dark:text-gray-100 font-jetbrains-mono">
              <Clock size={14} className="mr-2 text-[var(--brand)]" />
              <div>
                <div>{when}</div>
                {durationLabel ? (
                  <div className="mt-1 text-xs text-gray-600 dark:text-gray-300">
                    Duration: {durationLabel}
                  </div>
                ) : null}
              </div>
            </div>
          </div>

          <div className={boxClass}>
            <div className="text-xs text-gray-600 dark:text-gray-400 font-jetbrains-mono">
              Client
            </div>
            <div className="mt-1 flex items-start gap-2 text-sm text-gray-900 dark:text-gray-100 font-jetbrains-mono">
              <UserIcon size={14} className="mt-[2px] text-[var(--brand)]" />
              <div className="min-w-0">
                <div className="font-semibold truncate">
                  {event.client_name || "—"}
                </div>
                {clientPhone ? (
                  <div className="mt-1 flex items-center gap-2 text-xs text-gray-600 dark:text-gray-300">
                    <Phone size={12} />
                    <span className="truncate">{clientPhone}</span>
                  </div>
                ) : null}
                {clientEmail ? (
                  <div className="mt-1 flex items-center gap-2 text-xs text-gray-600 dark:text-gray-300">
                    <Mail size={12} />
                    <span className="truncate">{clientEmail}</span>
                  </div>
                ) : null}
              </div>
            </div>
          </div>

          {event.property_title ? (
            <div className={`sm:col-span-2 ${boxClass}`}>
              <div className="text-xs text-gray-600 dark:text-gray-400 font-jetbrains-mono">
                Property
              </div>
              <div className="mt-1 flex items-center gap-2 text-sm text-gray-900 dark:text-gray-100 font-jetbrains-mono">
                <Home size={14} className="text-[var(--brand)]" />
                <span className="truncate">{event.property_title}</span>
              </div>
            </div>
          ) : null}

          {event.notes ? (
            <div className={`sm:col-span-2 ${boxClass}`}>
              <div className="text-xs text-gray-600 dark:text-gray-400 font-jetbrains-mono">
                Notes
              </div>
              <div className="mt-1 text-sm text-gray-900 dark:text-gray-100 font-jetbrains-mono whitespace-pre-wrap">
                {String(event.notes)}
              </div>
            </div>
          ) : null}
        </div>

        <div className="mt-5 flex flex-col sm:flex-row gap-3 justify-end">
          <button
            type="button"
            onClick={() => onEdit?.(event)}
            className="inline-flex items-center justify-center gap-2 px-4 py-2 rounded-lg border border-black/10 dark:border-white/10 bg-white/60 dark:bg-white/5 hover:bg-white/80 dark:hover:bg-white/10 text-gray-900 dark:text-gray-100 font-jetbrains-mono transition-colors"
          >
            <Pencil size={16} />
            Edit
          </button>
          <button
            type="button"
            onClick={onClose}
            className="px-4 py-2 rounded-lg bg-[var(--brand90)] hover:bg-[var(--brand)] text-white font-medium font-jetbrains-mono transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </ModalShell>
  );
}
