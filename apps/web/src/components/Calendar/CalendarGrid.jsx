import {
  addDays,
  endOfMonth,
  endOfWeek,
  format,
  isSameDay,
  isSameMonth,
  isToday,
  startOfMonth,
  startOfWeek,
} from "date-fns";

export function CalendarGrid({
  monthDate,
  selectedDate,
  onSelectDate,
  eventsByDay,
}) {
  const start = startOfWeek(startOfMonth(monthDate), { weekStartsOn: 1 });
  const end = endOfWeek(endOfMonth(monthDate), { weekStartsOn: 1 });

  const days = [];
  let cur = start;
  while (cur <= end) {
    days.push(cur);
    cur = addDays(cur, 1);
  }

  const weeks = [];
  for (let i = 0; i < days.length; i += 7) {
    weeks.push(days.slice(i, i + 7));
  }

  // Week starts on Monday, but labels should be English.
  const weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

  return (
    <div>
      <div className="grid grid-cols-7 gap-1 sm:gap-2">
        {weekdays.map((w) => (
          <div
            key={w}
            className="text-[11px] text-gray-600 dark:text-gray-400 font-jetbrains-mono text-center"
          >
            {w}
          </div>
        ))}
      </div>

      <div className="mt-2 grid grid-cols-7 gap-1 sm:gap-2">
        {weeks.flat().map((d) => {
          const inMonth = isSameMonth(d, monthDate);
          const isSel = isSameDay(d, selectedDate);
          const isT = isToday(d);

          const key = format(d, "yyyy-MM-dd");
          const has = (eventsByDay.get(key) || []).length > 0;

          // Make the calendar fit on large screens by avoiding aspect-square.
          const baseClass =
            "w-full h-11 sm:h-12 lg:h-14 rounded-lg border text-sm font-jetbrains-mono flex flex-col items-center justify-center transition-colors";

          const borderClass = isSel
            ? "border-[var(--brand)]"
            : "border-black/10 dark:border-white/10";

          const bgClass = isSel
            ? "bg-[var(--brandSoft)] dark:bg-[var(--brandSoftDark)]"
            : "bg-white/60 dark:bg-white/5 hover:bg-white/80 dark:hover:bg-white/10";

          const textClass = !inMonth
            ? "text-gray-400 dark:text-gray-500"
            : "text-gray-900 dark:text-gray-100";

          const todayRing = isT && !isSel ? "ring-1 ring-[var(--brand)]" : "";

          return (
            <button
              key={key}
              type="button"
              onClick={() => onSelectDate(d)}
              className={`${baseClass} ${borderClass} ${bgClass} ${textClass} ${todayRing}`}
            >
              <div className="leading-none">{format(d, "d")}</div>
              <div className="mt-1 h-[6px]">
                {has ? (
                  <div className="h-[6px] w-[6px] rounded-full bg-[var(--brand)]"></div>
                ) : null}
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
