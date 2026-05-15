import sql from "@/app/api/utils/sql";

const UUID_RE =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

export function normalizeUserIdToUuid(rawUserId) {
  if (rawUserId === null || rawUserId === undefined) {
    return null;
  }

  const asString = String(rawUserId).trim();

  if (UUID_RE.test(asString)) {
    return asString.toLowerCase();
  }

  // Most Anything projects use integer auth user IDs (NextAuth default).
  // Our app DB schema uses UUIDs. So we map the integer ID to a stable UUID.
  // Example: 1 -> 00000000-0000-0000-0000-000000000001
  if (/^\d+$/.test(asString)) {
    const padded =
      asString.length <= 12 ? asString.padStart(12, "0") : asString.slice(-12);
    return `00000000-0000-0000-0000-${padded}`;
  }

  return null;
}

export async function getDbUserIdFromSession(session) {
  const dbUserId = normalizeUserIdToUuid(session?.user?.id);
  if (!dbUserId) {
    return null;
  }

  // Ensure the profile row exists to satisfy FK constraints.
  // Keep it additive: never overwrite existing profile data.
  try {
    await sql(
      "INSERT INTO profiles (id, full_name) VALUES ($1, $2) ON CONFLICT (id) DO NOTHING",
      [dbUserId, session?.user?.name || null],
    );
  } catch (error) {
    // If this fails, we still return dbUserId; routes will fail loudly anyway.
    console.error(
      "getDbUserIdFromSession: failed to ensure profiles row",
      error,
    );
  }

  return dbUserId;
}
