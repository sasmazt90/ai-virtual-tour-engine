export function dateKey(d) {
  try {
    // IMPORTANT: use local date parts (not toISOString) so events don’t jump days
    // for users in non-UTC timezones.
    const dt = new Date(d);
    if (Number.isNaN(dt.valueOf())) return "";

    const y = dt.getFullYear();
    const m = String(dt.getMonth() + 1).padStart(2, "0");
    const day = String(dt.getDate()).padStart(2, "0");
    return `${y}-${m}-${day}`;
  } catch {
    return "";
  }
}

export function toHHMM(date) {
  try {
    const h = String(date.getHours()).padStart(2, "0");
    const m = String(date.getMinutes()).padStart(2, "0");
    return `${h}:${m}`;
  } catch {
    return "09:00";
  }
}

export function humanMeetingType(ev) {
  const t = String(ev?.event_type || "");
  const ch = String(ev?.event_channel || "");

  // Ensure UI never shows Turkish labels.
  if (t === "visit") return "Property visit";

  if (ch === "email") return "Email";
  if (ch === "in_person") return "In-person meeting";
  return "Phone call";
}

export function eventBadge(ev) {
  const t = String(ev?.event_type || "");
  if (t === "visit") {
    return {
      text: "visit",
      className:
        "bg-green-100 dark:bg-green-900 text-green-800 dark:text-green-200",
    };
  }
  return {
    text: "meeting",
    className: "bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200",
  };
}
