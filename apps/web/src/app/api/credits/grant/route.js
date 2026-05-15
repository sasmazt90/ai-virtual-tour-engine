import sql from "@/app/api/utils/sql";
import { auth } from "@/auth";
import {
  getDbUserIdFromSession,
  normalizeUserIdToUuid,
} from "@/app/api/utils/dbUser";

const UUID_RE =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

export async function POST(request) {
  try {
    const session = await auth();
    if (!session || !session.user?.id) {
      return Response.json({ error: "Unauthorized" }, { status: 401 });
    }

    // Safety: only allow the site owner to grant credits.
    const requesterEmail = String(session.user?.email || "").toLowerCase();
    const isAdmin = requesterEmail === "sasmazt90@gmail.com";
    if (!isAdmin) {
      return Response.json({ error: "Forbidden" }, { status: 403 });
    }

    const body = await request.json().catch(() => ({}));
    const amount = Number(body?.credits || 0);

    if (!Number.isFinite(amount) || amount <= 0) {
      return Response.json(
        { error: "Invalid credits amount" },
        { status: 400 },
      );
    }

    // Admin can grant to any user. Default: grant to self.
    const targetRaw = body?.targetUserId ?? null;

    const requesterUserId = await getDbUserIdFromSession(session);
    if (!requesterUserId) {
      return Response.json({ error: "Unauthorized" }, { status: 401 });
    }

    const targetUserId = targetRaw
      ? normalizeUserIdToUuid(targetRaw)
      : requesterUserId;

    if (!targetUserId || !UUID_RE.test(String(targetUserId))) {
      return Response.json({ error: "Invalid target user" }, { status: 400 });
    }

    const reason = body?.reason ? String(body.reason).slice(0, 200) : "manual";

    const [walletRows] = await sql.transaction((txn) => [
      txn(
        `
        INSERT INTO credits_wallet (user_id, balance_credits)
        VALUES ($1, $2)
        ON CONFLICT (user_id)
        DO UPDATE SET
          balance_credits = credits_wallet.balance_credits + EXCLUDED.balance_credits,
          updated_at = NOW()
        RETURNING user_id, balance_credits, updated_at
        `,
        [targetUserId, Math.trunc(amount)],
      ),
      txn(
        `
        INSERT INTO credit_transactions (
          user_id,
          transaction_type,
          credits_delta,
          provider,
          meta
        )
        VALUES ($1, 'adjustment', $2, 'manual', $3::jsonb)
        `,
        [
          targetUserId,
          Math.trunc(amount),
          JSON.stringify({
            reason,
            kind: "manual_grant",
            grantedBy: requesterEmail,
          }),
        ],
      ),
    ]);

    const wallet = walletRows?.[0];

    return Response.json({
      success: true,
      user_id: wallet?.user_id,
      balance: Number(wallet?.balance_credits || 0),
      updated_at: wallet?.updated_at,
    });
  } catch (error) {
    console.error("POST /api/credits/grant error:", error);
    return Response.json({ error: "Internal Server Error" }, { status: 500 });
  }
}
