import { useState, useCallback, useMemo } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { dateKey, toHHMM } from "@/utils/calendarHelpers";

function toDateInputValue(d) {
  // Use local date key (not UTC) so date inputs match what the user sees.
  return dateKey(d);
}

function parseYMD(dateStr) {
  const s = String(dateStr || "").trim();
  const m = s.match(/^(\d{4})-(\d{2})-(\d{2})$/);
  if (!m) return null;

  const y = Number(m[1]);
  const mo = Number(m[2]);
  const d = Number(m[3]);

  if (!Number.isFinite(y) || !Number.isFinite(mo) || !Number.isFinite(d)) {
    return null;
  }
  if (mo < 1 || mo > 12) return null;
  if (d < 1 || d > 31) return null;

  return { y, mo, d };
}

function datePartsFromDate(value) {
  try {
    const dt = new Date(value);
    if (Number.isNaN(dt.valueOf())) return null;
    return { y: dt.getFullYear(), mo: dt.getMonth() + 1, d: dt.getDate() };
  } catch {
    return null;
  }
}

function buildLocalDateFromParts(parts, hh, mm) {
  const local = new Date(parts.y, parts.mo - 1, parts.d, hh, mm, 0, 0);
  if (Number.isNaN(local.valueOf())) return null;
  return local;
}

function parseHHMM(timeHHMM) {
  const s = String(timeHHMM || "").trim();
  if (!/^[0-2]\d:[0-5]\d$/.test(s)) return null;
  const [hhStr, mmStr] = s.split(":");
  const hh = Number(hhStr);
  const mm = Number(mmStr);
  if (!Number.isFinite(hh) || !Number.isFinite(mm)) return null;
  if (hh > 23 || mm > 59) return null;
  return { hh, mm };
}

function buildStartsAtIso({ dateStr, timeStr, fallbackDate }) {
  const parts = parseYMD(dateStr) || datePartsFromDate(fallbackDate);
  if (!parts) return null;

  const parsed = parseHHMM(timeStr);
  if (!parsed) return null;

  // CRITICAL: build local Date via numeric constructor (avoids browser differences).
  const local = buildLocalDateFromParts(parts, parsed.hh, parsed.mm);
  if (!local) return null;

  return local.toISOString();
}

function uiMeetingTypeFromEvent(ev) {
  const type = String(ev?.event_type || "");
  const ch = String(ev?.event_channel || "");

  // Keep UI meeting type codes in English.
  if (type === "visit") return "property_visit";
  if (ch === "email") return "email";
  if (ch === "in_person") return "in_person";
  return "phone";
}

function splitNotesForUi(ev) {
  const notes = ev?.notes ? String(ev.notes) : "";
  if (!notes) return { address: "", notes: "" };

  const lines = notes.split("\n");
  const first = String(lines[0] || "").trim();

  if (first.toLowerCase().startsWith("address:")) {
    const address = first.slice("address:".length).trim();
    const rest = lines.slice(1).join("\n").trim();
    return { address, notes: rest };
  }

  return { address: "", notes };
}

export function useCreateEvent(userId, selectedDate) {
  const queryClient = useQueryClient();

  const [createOpen, setCreateOpen] = useState(false);
  const [mode, setMode] = useState("create"); // create | edit
  const [editingId, setEditingId] = useState(null);

  const [formDate, setFormDate] = useState(""); // yyyy-mm-dd
  const [formClientId, setFormClientId] = useState("");
  const [formType, setFormType] = useState("phone");
  const [formTime, setFormTime] = useState("09:00");
  const [formDuration, setFormDuration] = useState("60");
  const [formPropertyId, setFormPropertyId] = useState("");
  const [formAddress, setFormAddress] = useState("");
  const [formNotes, setFormNotes] = useState("");
  const [formError, setFormError] = useState(null);

  const safeDate = useMemo(() => {
    // Always compute a local midnight date from the date input.
    // (This avoids Date("YYYY-MM-DD") parsing differences.)
    const parts = parseYMD(formDate) || datePartsFromDate(selectedDate) || null;
    if (!parts) return new Date();

    const localMidnight = buildLocalDateFromParts(parts, 0, 0);
    return localMidnight || new Date();
  }, [formDate, selectedDate]);

  const saveMutation = useMutation({
    mutationFn: async () => {
      const startsAtIso = buildStartsAtIso({
        dateStr: formDate,
        timeStr: formTime,
        fallbackDate: selectedDate,
      });

      if (!startsAtIso) {
        throw new Error("Please enter a valid date and time");
      }

      const payload = {
        id: mode === "edit" ? editingId : undefined,
        clientId: formClientId,
        meetingType: formType,
        // NEW: send the fully-computed local datetime as an ISO string.
        // Server must store it as-is (no timezone rebuilding).
        startsAtIso,
        durationMinutes: Number(formDuration || 60),
        propertyId: formType === "property_visit" ? formPropertyId : null,
        address: formType === "in_person" ? formAddress : null,
        notes: formNotes,

        // Backward compat (ignored by server when startsAtIso is present)
        startsAt: safeDate.toISOString(),
        time: formTime,
      };

      const res = await fetch("/api/calendar/events", {
        method: mode === "edit" ? "PUT" : "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body?.error || "Could not save event");
      }

      return res.json();
    },
    onSuccess: async () => {
      await queryClient.invalidateQueries({
        queryKey: ["calendar-events", userId],
      });
      setCreateOpen(false);
      setFormError(null);
      setEditingId(null);
      setMode("create");
      setFormClientId("");
      setFormType("phone");
      setFormTime("09:00");
      setFormDuration("60");
      setFormPropertyId("");
      setFormAddress("");
      setFormNotes("");
    },
    onError: (e) => {
      console.error(e);
      setFormError(e?.message || "Could not save event");
    },
  });

  const deleteMutation = useMutation({
    mutationFn: async (id) => {
      const res = await fetch("/api/calendar/events", {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id }),
      });

      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body?.error || "Could not delete event");
      }

      return res.json();
    },
    onSuccess: async () => {
      await queryClient.invalidateQueries({
        queryKey: ["calendar-events", userId],
      });
      setCreateOpen(false);
      setFormError(null);
      setEditingId(null);
      setMode("create");
    },
    onError: (e) => {
      console.error(e);
      setFormError(e?.message || "Could not delete event");
    },
  });

  const onOpenCreate = useCallback(() => {
    setFormError(null);
    setMode("create");
    setEditingId(null);
    setFormDate(toDateInputValue(selectedDate || new Date()));
    setFormTime(toHHMM(new Date()));
    setFormDuration("60");
    setCreateOpen(true);
  }, [selectedDate]);

  const onOpenEdit = useCallback((ev) => {
    if (!ev?.id) return;

    setFormError(null);
    setMode("edit");
    setEditingId(ev.id);

    setFormDate(toDateInputValue(ev.starts_at));
    setFormTime(toHHMM(new Date(ev.starts_at)));

    const durMin = Math.max(
      5,
      Math.round((new Date(ev.ends_at) - new Date(ev.starts_at)) / 60000),
    );
    setFormDuration(String(Number.isFinite(durMin) ? durMin : 60));

    setFormClientId(ev.client_id ? String(ev.client_id) : "");
    setFormType(uiMeetingTypeFromEvent(ev));
    setFormPropertyId(ev.property_id ? String(ev.property_id) : "");

    const parsed = splitNotesForUi(ev);
    setFormAddress(parsed.address);
    setFormNotes(parsed.notes);

    setCreateOpen(true);
  }, []);

  const onCloseCreate = useCallback(() => {
    setCreateOpen(false);
  }, []);

  const onSubmit = useCallback(() => {
    setFormError(null);
    saveMutation.mutate();
  }, [saveMutation]);

  const onDelete = useCallback(() => {
    setFormError(null);
    if (!editingId) {
      setFormError("Event id missing");
      return;
    }
    deleteMutation.mutate(editingId);
  }, [deleteMutation, editingId]);

  return {
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
    isSubmitting: saveMutation.isPending,
    isDeleting: deleteMutation.isPending,
  };
}
