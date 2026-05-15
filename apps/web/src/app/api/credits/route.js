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

    const rows = await sql(
      "SELECT user_id, balance_credits, updated_at FROM credits_wallet WHERE user_id = $1 LIMIT 1",
      [userId],
    );

    const walletExists = rows.length > 0;

    if (!walletExists) {
      // Additive: expose whether the wallet row exists so the UI can show a health indicator.
      return Response.json({
        balance: 0,
        updated_at: new Date().toISOString(),
        walletExists: false,
        walletInconsistent: false,
      });
    }

    const rawBalance = rows[0]?.balance_credits;
    const balance = Number(rawBalance);

    // Should never happen due to DB constraint, but keep the status logic honest.
    const walletInconsistent =
      rawBalance === null || !Number.isFinite(balance) || balance < 0;

    return Response.json({
      balance: Number.isFinite(balance) ? balance : 0,
      updated_at: rows[0].updated_at,
      walletExists: true,
      walletInconsistent,
    });
  } catch (error) {
    console.error("GET /api/credits error:", error);
    return Response.json({ error: "Internal Server Error" }, { status: 500 });
  }
}
