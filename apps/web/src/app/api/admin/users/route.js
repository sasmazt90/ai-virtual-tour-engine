import sql from "@/app/api/utils/sql";
import { auth } from "@/auth";
import { normalizeUserIdToUuid } from "@/app/api/utils/dbUser";
import { isCreditAdminEmail } from "@/app/api/utils/pricing";

export async function GET() {
  try {
    const session = await auth();
    if (!session || !session.user?.id) {
      return Response.json({ error: "Unauthorized" }, { status: 401 });
    }

    const requesterEmail = String(session.user?.email || "").toLowerCase();
    if (!isCreditAdminEmail(requesterEmail)) {
      return Response.json({ error: "Forbidden" }, { status: 403 });
    }

    const rows = await sql(
      `
      SELECT id, email, name
      FROM auth_users
      WHERE email IS NOT NULL
      ORDER BY COALESCE(name, ''), email
      `,
      [],
    );

    const users = rows
      .map((u) => {
        const dbUserId = normalizeUserIdToUuid(u.id);
        return {
          user_id: dbUserId,
          auth_user_id: u.id,
          email: u.email,
          name: u.name,
        };
      })
      .filter((u) => !!u.user_id);

    return Response.json(users);
  } catch (error) {
    console.error("GET /api/admin/users error:", error);
    return Response.json({ error: "Internal Server Error" }, { status: 500 });
  }
}
