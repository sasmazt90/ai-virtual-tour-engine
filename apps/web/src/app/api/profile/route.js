import sql from "@/app/api/utils/sql";
import { auth } from "@/auth";
import { getDbUserIdFromSession } from "@/app/api/utils/dbUser";

export async function GET() {
  try {
    const session = await auth();
    if (!session || !session.user?.id) {
      return Response.json({ error: "Unauthorized" }, { status: 401 });
    }

    const userId = await getDbUserIdFromSession(session);
    if (!userId) {
      return Response.json({ error: "Unauthorized" }, { status: 401 });
    }

    const profiles = await sql`
      SELECT * FROM profiles WHERE id = ${userId} LIMIT 1
    `;

    if (profiles.length === 0) {
      return Response.json({
        full_name: session.user.name || null,
        company: null,
        company_logo_url: null,
      });
    }

    return Response.json(profiles[0]);
  } catch (error) {
    console.error("GET /api/profile error:", error);
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
    const { full_name, company, company_logo_url } = body;

    const result = await sql`
      UPDATE profiles 
      SET 
        full_name = COALESCE(${full_name}, full_name),
        company = COALESCE(${company}, company),
        company_logo_url = COALESCE(${company_logo_url}, company_logo_url)
      WHERE id = ${userId}
      RETURNING *
    `;

    if (result.length > 0) {
      return Response.json(result[0]);
    }

    const inserted = await sql`
      INSERT INTO profiles (id, full_name, company, company_logo_url)
      VALUES (${userId}, ${full_name}, ${company}, ${company_logo_url})
      RETURNING *
    `;

    return Response.json(inserted[0]);
  } catch (error) {
    console.error("PUT /api/profile error:", error);
    return Response.json({ error: "Internal Server Error" }, { status: 500 });
  }
}
