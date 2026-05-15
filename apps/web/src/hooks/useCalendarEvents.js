import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { dateKey } from "@/utils/calendarHelpers";

export function useCalendarEvents(userId, options) {
  const clientId = options?.clientId ? String(options.clientId) : null;

  const { data: events, isLoading } = useQuery({
    queryKey: ["calendar-events", userId, clientId],
    queryFn: async () => {
      const params = new URLSearchParams();
      if (clientId) params.set("clientId", clientId);

      const url = params.toString()
        ? `/api/calendar/events?${params.toString()}`
        : "/api/calendar/events";

      const res = await fetch(url);
      if (!res.ok) throw new Error("Failed to fetch events");
      return res.json();
    },
    enabled: !!userId,
  });

  const eventsArr = useMemo(() => {
    return Array.isArray(events) ? events : [];
  }, [events]);

  const eventsByDay = useMemo(() => {
    const map = new Map();
    for (const ev of eventsArr) {
      const k = dateKey(ev?.starts_at);
      if (!k) continue;
      const prev = map.get(k) || [];
      prev.push(ev);
      map.set(k, prev);
    }

    for (const [k, list] of map.entries()) {
      list.sort((a, b) => new Date(a.starts_at) - new Date(b.starts_at));
      map.set(k, list);
    }

    return map;
  }, [eventsArr]);

  const upcomingEvents = useMemo(() => {
    const now = new Date();
    return eventsArr
      .filter((event) => {
        const eventDate = new Date(event.starts_at);
        return eventDate >= now;
      })
      .sort((a, b) => new Date(a.starts_at) - new Date(b.starts_at));
  }, [eventsArr]);

  return {
    events: eventsArr,
    eventsByDay,
    upcomingEvents,
    isLoading,
  };
}
