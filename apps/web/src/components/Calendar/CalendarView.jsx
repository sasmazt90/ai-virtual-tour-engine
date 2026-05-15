import { useMemo } from "react";
import { format } from "date-fns";
import {
  Calendar as CalendarIcon,
  ChevronLeft,
  ChevronRight,
} from "lucide-react";
import { CalendarGrid } from "./CalendarGrid";
import { EventCard } from "./EventCard";
import { dateKey } from "@/utils/calendarHelpers";

export function CalendarView({
  monthDate,
  selectedDate,
  onSelectDate,
  eventsByDay,
  onPrevMonth,
  onNextMonth,
  isLoading,
  onEventClick,
}) {
  const selectedKey = useMemo(() => dateKey(selectedDate), [selectedDate]);

  const selectedEvents = useMemo(() => {
    return eventsByDay.get(selectedKey) || [];
  }, [eventsByDay, selectedKey]);

  return (
    <div className="rounded-2xl border border-black/10 dark:border-white/10 bg-white/70 dark:bg-white/5 backdrop-blur shadow-[0_14px_60px_rgba(0,0,0,0.18)] p-4 sm:p-6">
      <div className="flex items-center justify-between gap-3 mb-4">
        <div className="flex items-center gap-2">
          <CalendarIcon className="text-[var(--brand)]" size={20} />
          <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-50 font-jetbrains-mono">
            Calendar View
          </h3>
        </div>

        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={onPrevMonth}
            className="p-2 rounded-lg border border-black/10 dark:border-white/10 bg-white/60 dark:bg-white/5 hover:bg-white/80 dark:hover:bg-white/10 transition-colors"
            aria-label="Previous month"
          >
            <ChevronLeft
              size={16}
              className="text-gray-800 dark:text-gray-100"
            />
          </button>
          <div className="text-sm text-gray-700 dark:text-gray-200 font-jetbrains-mono min-w-[120px] text-center">
            {format(monthDate, "MMMM yyyy")}
          </div>
          <button
            type="button"
            onClick={onNextMonth}
            className="p-2 rounded-lg border border-black/10 dark:border-white/10 bg-white/60 dark:bg-white/5 hover:bg-white/80 dark:hover:bg-white/10 transition-colors"
            aria-label="Next month"
          >
            <ChevronRight
              size={16}
              className="text-gray-800 dark:text-gray-100"
            />
          </button>
        </div>
      </div>

      <CalendarGrid
        monthDate={monthDate}
        selectedDate={selectedDate}
        onSelectDate={onSelectDate}
        eventsByDay={eventsByDay}
      />

      <div className="mt-6 border-t border-black/10 dark:border-white/10 pt-4">
        <div className="text-sm font-semibold text-gray-900 dark:text-gray-50 font-jetbrains-mono">
          Events on {selectedKey}
        </div>

        {isLoading ? (
          <div className="mt-3 text-sm text-gray-700 dark:text-gray-300 font-jetbrains-mono">
            Loading events...
          </div>
        ) : selectedEvents.length > 0 ? (
          <div className="mt-3 space-y-3">
            {selectedEvents.map((event) => (
              <EventCard key={event.id} event={event} onClick={onEventClick} />
            ))}
          </div>
        ) : (
          <div className="mt-3 text-sm text-gray-600 dark:text-gray-300 font-jetbrains-mono">
            No events on this day.
          </div>
        )}
      </div>
    </div>
  );
}
