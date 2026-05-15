import sql from "@/app/api/utils/sql";
import { auth } from "@/auth";
import { getDbUserIdFromSession } from "@/app/api/utils/dbUser";

// Keep UI meeting type codes in English.
const TYPE_OPTIONS = ["phone", "email", "property_visit", "in_person"];

function buildEventMapping(uiType) {
  const t = String(uiType || "").toLowerCase();

  // DB enums:
  // event_type: meeting | visit
  // event_channel: phone | email | in_person
  if (t === "property_visit") {
    return { event_type: "visit", event_channel: "in_person" };
  }

  if (t === "in_person") {
    return { event_type: "meeting", event_channel: "in_person" };
  }

  if (t === "email") {
    return { event_type: "meeting", event_channel: "email" };
  }

  // default: phone
  return { event_type: "meeting", event_channel: "phone" };
}

function parseHHMM(timeHHMM) {
  const timeStr = String(timeHHMM || "").trim();
  if (!/^[0-2]\d:[0-5]\d$/.test(timeStr)) {
    return null;
  }
  const parts = timeStr.split(":");
  const hh = Number(parts[0]);
  const mm = Number(parts[1]);
  if (hh > 23 || mm > 59) {
    return null;
  }
  return { hh, mm };
}

function normalizeDurationMinutes(value) {
  const n = Math.trunc(Number(value));
  if (!Number.isFinite(n)) return 60;
  if (n < 5) return 60;
  if (n > 24 * 60) return 24 * 60;
  return n;
}

function buildNotes(mapped, address, notes) {
  const notesParts = [];
  if (mapped.event_channel === "in_person" && address) {
    notesParts.push(`Address: ${String(address).trim()}`);
  }
  if (notes) {
    notesParts.push(String(notes).trim());
  }
  return notesParts.length > 0 ? notesParts.join("\n\n") : null;
}

export async function GET(request) {
  try {
    const session = await auth();
    if (!session || !session.user?.id) {
      return Response.json({ error: "Unauthorized" }, { status: 401 });
    }

    const userId = await getDbUserIdFromSession(session);
    if (!userId) {
      return Response.json({ error: "Unauthorized" }, { status: 401 });
    }

    const { searchParams } = new URL(request.url);
    const clientId = searchParams.get("clientId");

    let queryStr = `
      SELECT 
        ce.*, 
        c.full_name as client_name,
        c.phone as client_phone,
        c.email as client_email,
        p.title as property_title
      FROM calendar_events ce
      LEFT JOIN clients c ON ce.client_id = c.id
      LEFT JOIN properties p ON ce.property_id = p.id
      WHERE ce.user_id = $1
    `;

    const values = [userId];
    let paramIndex = 2;

    if (clientId) {
      queryStr += ` AND ce.client_id = $${paramIndex}`;
      values.push(clientId);
      paramIndex++;
    }

    queryStr += ` ORDER BY ce.starts_at ASC`;

    const events = await sql(queryStr, values);
    return Response.json(events);
  } catch (error) {
    console.error("GET /api/calendar/events error:", error);
    return Response.json({ error: "Internal Server Error" }, { status: 500 });
  }
}

export async function POST(request) {
  try {
    const session = await auth();
    if (!session || !session.user?.id) {
      return Response.json({ error: "Unauthorized" }, { status: 401 });
    }

    const userId = await getDbUserIdFromSession(session);
    if (!userId) {
      return Response.json({ error: "Unauthorized" }, { status: 401 });
    }

    const body = await request.json().catch(() => ({}));

    const clientId = body?.clientId || null;
    const uiType = body?.meetingType || null;

    // NEW: browser sends a full ISO datetime already (local date + time combined).
    const startsAtIso = body?.startsAtIso || null;

    // Backward compat fields
    const startsAt = body?.startsAt || null;
    const timeHHMM = body?.time || null;

    const durationMinutes = normalizeDurationMinutes(body?.durationMinutes);
    const propertyId = body?.propertyId || null;
    const address = body?.address || null;
    const notes = body?.notes || null;

    if (!clientId) {
      return Response.json({ error: "Client is required" }, { status: 400 });
    }

    if (!uiType || !TYPE_OPTIONS.includes(String(uiType).toLowerCase())) {
      return Response.json(
        { error: "Meeting type is required" },
        { status: 400 },
      );
    }

    let starts = null;

    if (startsAtIso) {
      const parsedStart = new Date(startsAtIso);
      if (Number.isNaN(parsedStart.valueOf())) {
        return Response.json({ error: "Invalid date/time" }, { status: 400 });
      }
      starts = parsedStart;
    } else {
      // Legacy path: startsAt is a date and timeHHMM provides hours/minutes.
      const start = new Date(startsAt);
      if (!startsAt || Number.isNaN(start.valueOf())) {
        return Response.json({ error: "Invalid date" }, { status: 400 });
      }

      const parsed = parseHHMM(timeHHMM);
      if (!parsed) {
        return Response.json({ error: "Invalid time" }, { status: 400 });
      }

      starts = new Date(start);
      starts.setHours(parsed.hh, parsed.mm, 0, 0);
    }

    const ends = new Date(starts.getTime() + durationMinutes * 60 * 1000);

    const mapped = buildEventMapping(uiType);

    if (mapped.event_type === "visit" && !propertyId) {
      return Response.json(
        { error: "Property is required for a visit" },
        { status: 400 },
      );
    }

    const finalNotes = buildNotes(mapped, address, notes);

    const inserted = await sql(
      `
      INSERT INTO calendar_events (
        user_id,
        property_id,
        client_id,
        event_type,
        event_channel,
        starts_at,
        ends_at,
        notes
      )
      VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
      RETURNING *
      `,
      [
        userId,
        mapped.event_type === "visit" ? propertyId : null,
        clientId,
        mapped.event_type,
        mapped.event_channel,
        starts.toISOString(),
        ends.toISOString(),
        finalNotes,
      ],
    );

    return Response.json(inserted[0], { status: 201 });
  } catch (error) {
    console.error("POST /api/calendar/events error:", error);
    return Response.json({ error: "Internal Server Error" }, { status: 500 });
  }
}

export async function PUT(request) {
  try {
    const session = await auth();
    if (!session || !session.user?.id) {
      return Response.json({ error: "Unauthorized" }, { status: 401 });
    }

    const userId = await getDbUserIdFromSession(session);
    if (!userId) {
      return Response.json({ error: "Unauthorized" }, { status: 401 });
    }

    const body = await request.json().catch(() => ({}));

    const id = body?.id || null;
    const clientId = body?.clientId || null;
    const uiType = body?.meetingType || null;

    const startsAtIso = body?.startsAtIso || null;
    const startsAt = body?.startsAt || null;
    const timeHHMM = body?.time || null;

    const durationMinutes = normalizeDurationMinutes(body?.durationMinutes);
    const propertyId = body?.propertyId || null;
    const address = body?.address || null;
    const notes = body?.notes || null;

    if (!id) {
      return Response.json({ error: "Event id is required" }, { status: 400 });
    }

    if (!clientId) {
      return Response.json({ error: "Client is required" }, { status: 400 });
    }

    if (!uiType || !TYPE_OPTIONS.includes(String(uiType).toLowerCase())) {
      return Response.json(
        { error: "Meeting type is required" },
        { status: 400 },
      );
    }

    let starts = null;

    if (startsAtIso) {
      const parsedStart = new Date(startsAtIso);
      if (Number.isNaN(parsedStart.valueOf())) {
        return Response.json({ error: "Invalid date/time" }, { status: 400 });
      }
      starts = parsedStart;
    } else {
      const start = new Date(startsAt);
      if (!startsAt || Number.isNaN(start.valueOf())) {
        return Response.json({ error: "Invalid date" }, { status: 400 });
      }

      const parsed = parseHHMM(timeHHMM);
      if (!parsed) {
        return Response.json({ error: "Invalid time" }, { status: 400 });
      }

      starts = new Date(start);
      starts.setHours(parsed.hh, parsed.mm, 0, 0);
    }

    const ends = new Date(starts.getTime() + durationMinutes * 60 * 1000);

    const mapped = buildEventMapping(uiType);

    if (mapped.event_type === "visit" && !propertyId) {
      return Response.json(
        { error: "Property is required for a visit" },
        { status: 400 },
      );
    }

    const finalNotes = buildNotes(mapped, address, notes);

    const updatedRows = await sql(
      `
      UPDATE calendar_events
      SET
        property_id = $1,
        client_id = $2,
        event_type = $3,
        event_channel = $4,
        starts_at = $5,
        ends_at = $6,
        notes = $7
      WHERE id = $8 AND user_id = $9
      RETURNING *
      `,
      [
        mapped.event_type === "visit" ? propertyId : null,
        clientId,
        mapped.event_type,
        mapped.event_channel,
        starts.toISOString(),
        ends.toISOString(),
        finalNotes,
        id,
        userId,
      ],
    );

    if (updatedRows.length === 0) {
      return Response.json({ error: "Not found" }, { status: 404 });
    }

    return Response.json(updatedRows[0]);
  } catch (error) {
    console.error("PUT /api/calendar/events error:", error);
    return Response.json({ error: "Internal Server Error" }, { status: 500 });
  }
}

export async function DELETE(request) {
  try {
    const session = await auth();
    if (!session || !session.user?.id) {
      return Response.json({ error: "Unauthorized" }, { status: 401 });
    }

    const userId = await getDbUserIdFromSession(session);
    if (!userId) {
      return Response.json({ error: "Unauthorized" }, { status: 401 });
    }

    const body = await request.json().catch(() => ({}));
    const id = body?.id || null;

    if (!id) {
      return Response.json({ error: "Event id is required" }, { status: 400 });
    }

    const deleted = await sql(
      `DELETE FROM calendar_events WHERE id = $1 AND user_id = $2 RETURNING id`,
      [id, userId],
    );

    if (deleted.length === 0) {
      return Response.json({ error: "Not found" }, { status: 404 });
    }

    return Response.json({ success: true });
  } catch (error) {
    console.error("DELETE /api/calendar/events error:", error);
    return Response.json({ error: "Internal Server Error" }, { status: 500 });
  }
}
