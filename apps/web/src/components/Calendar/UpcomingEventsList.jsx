import { UpcomingEventCard } from "./UpcomingEventCard";

export function UpcomingEventsList({ events, isLoading, onEventClick }) {
  return (
    <div>
      <h2 className="text-xl font-semibold text-gray-50 mb-4 font-jetbrains-mono">
        Upcoming Events
      </h2>
      <div className="space-y-4">
        {isLoading ? (
          <p className="text-gray-300 font-jetbrains-mono">Loading events...</p>
        ) : events.length > 0 ? (
          events
            .slice(0, 12)
            .map((event) => (
              <UpcomingEventCard
                key={event.id}
                event={event}
                onClick={onEventClick}
              />
            ))
        ) : (
          <div className="rounded-2xl border border-white/10 bg-white/5 backdrop-blur p-6 text-center">
            <p className="text-gray-300 font-jetbrains-mono">
              No upcoming events
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
