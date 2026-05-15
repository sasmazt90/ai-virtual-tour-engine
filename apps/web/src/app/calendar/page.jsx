import { Header } from "@/components/Header";
import { useState, useCallback, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import useUser from "@/utils/useUser";
import { Plus } from "lucide-react";
import { addMonths, isSameMonth, subMonths } from "date-fns";
import { useCalendarEvents } from "@/hooks/useCalendarEvents";
import { useCreateEvent } from "@/hooks/useCreateEvent";
import { useEventDetail } from "@/hooks/useEventDetail";
import { CalendarView } from "@/components/Calendar/CalendarView";
import { UpcomingEventsList } from "@/components/Calendar/UpcomingEventsList";
import { CreateEventModal } from "@/components/Calendar/CreateEventModal";
import { EventDetailModal } from "@/components/Calendar/EventDetailModal";

export default function CalendarPage() {
  const { data: user, loading: userLoading } = useUser();
  const [selectedDate, setSelectedDate] = useState(new Date());
  const [monthDate, setMonthDate] = useState(new Date());

  useEffect(() => {
    // Keep month in sync when user picks a date outside current month
    if (!isSameMonth(selectedDate, monthDate)) {
      setMonthDate(selectedDate);
    }
  }, [selectedDate, monthDate]);

  const { eventsByDay, upcomingEvents, isLoading } = useCalendarEvents(
    user?.id,
  );

  const { data: clients } = useQuery({
    queryKey: ["clients", user?.id, "calendar"],
    queryFn: async () => {
      const res = await fetch("/api/clients?type=all");
      if (!res.ok) {
        throw new Error("Failed to fetch clients");
      }
      return res.json();
    },
    enabled: !!user?.id,
  });

  const { data: properties } = useQuery({
    queryKey: ["properties", user?.id, "calendar"],
    queryFn: async () => {
      const res = await fetch("/api/properties?status=all");
      if (!res.ok) {
        throw new Error("Failed to fetch properties");
      }
      return res.json();
    },
    enabled: !!user?.id,
  });

  const {
    createOpen,
    mode,
    onOpenEdit,
    formDate,
    setFormDate,
    formClientId,
    setFormClientId,
    formType,
    setFormType,
    formTime,
    setFormTime,
    formDuration,
    setFormDuration,
    formPropertyId,
    setFormPropertyId,
    formAddress,
    setFormAddress,
    formNotes,
    setFormNotes,
    formError,
    onOpenCreate,
    onCloseCreate,
    onSubmit,
    onDelete,
    isSubmitting,
    isDeleting,
  } = useCreateEvent(user?.id, selectedDate);

  const { detailOpen, detailEvent, onOpenDetail, onCloseDetail } =
    useEventDetail();

  const handleEditFromDetail = useCallback(
    (ev) => {
      onCloseDetail();
      onOpenEdit(ev);
    },
    [onCloseDetail, onOpenEdit],
  );

  const onPrevMonth = useCallback(() => {
    setMonthDate((d) => subMonths(d, 1));
  }, []);

  const onNextMonth = useCallback(() => {
    setMonthDate((d) => addMonths(d, 1));
  }, []);

  if (userLoading) {
    return (
      <div className="min-h-screen">
        <Header />
        <div className="pt-16 flex items-center justify-center min-h-screen">
          <p className="text-gray-700 dark:text-gray-300 font-jetbrains-mono">
            Loading...
          </p>
        </div>
      </div>
    );
  }

  if (!user) {
    if (typeof window !== "undefined") {
      window.location.href = "/account/signin";
    }
    return null;
  }

  return (
    <div className="min-h-screen">
      <Header />

      <div className="pt-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-8 py-8 sm:py-12">
          <div className="mb-8 flex flex-col sm:flex-row sm:items-end sm:justify-between gap-4">
            <div>
              <h1 className="text-3xl sm:text-4xl font-bold text-gray-900 dark:text-gray-50 mb-2 font-jetbrains-mono">
                Calendar
              </h1>
              <p className="text-gray-600 dark:text-gray-300 font-jetbrains-mono">
                Manage your meetings and property visits
              </p>
            </div>

            <button
              type="button"
              onClick={onOpenCreate}
              className="inline-flex items-center justify-center gap-2 rounded-lg bg-[var(--brand90)] hover:bg-[var(--brand)] text-white px-4 py-2 font-medium font-jetbrains-mono transition-colors"
            >
              <Plus size={18} />
              Add meeting
            </button>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            {/* Calendar View */}
            <div className="lg:col-span-2 space-y-6">
              <CalendarView
                monthDate={monthDate}
                selectedDate={selectedDate}
                onSelectDate={setSelectedDate}
                eventsByDay={eventsByDay}
                onPrevMonth={onPrevMonth}
                onNextMonth={onNextMonth}
                isLoading={isLoading}
                onEventClick={onOpenDetail}
              />
            </div>

            {/* Upcoming Events */}
            <div>
              <UpcomingEventsList
                events={upcomingEvents}
                isLoading={isLoading}
                onEventClick={onOpenDetail}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Create Event Modal */}
      <CreateEventModal
        open={createOpen}
        mode={mode}
        onClose={onCloseCreate}
        onDelete={onDelete}
        isDeleting={isDeleting}
        selectedDate={selectedDate}
        formDate={formDate}
        setFormDate={setFormDate}
        formClientId={formClientId}
        setFormClientId={setFormClientId}
        formType={formType}
        setFormType={setFormType}
        formTime={formTime}
        setFormTime={setFormTime}
        formDuration={formDuration}
        setFormDuration={setFormDuration}
        formPropertyId={formPropertyId}
        setFormPropertyId={setFormPropertyId}
        formAddress={formAddress}
        setFormAddress={setFormAddress}
        formNotes={formNotes}
        setFormNotes={setFormNotes}
        formError={formError}
        clients={clients}
        properties={properties}
        onSubmit={onSubmit}
        isSubmitting={isSubmitting}
      />

      {/* Detail Modal */}
      <EventDetailModal
        open={detailOpen}
        event={detailEvent}
        onClose={onCloseDetail}
        onEdit={handleEditFromDetail}
      />
    </div>
  );
}
