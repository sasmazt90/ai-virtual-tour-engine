import sql from "@/app/api/utils/sql";
import { auth } from "@/auth";
import {
  getDbUserIdFromSession,
  normalizeUserIdToUuid,
} from "@/app/api/utils/dbUser";
import { CREDIT_PACKS } from "@/app/api/utils/creditPacks";

function normalizePackType(raw) {
  const t = String(raw || "").toUpperCase();
  if (t === "BRONZE" || t === "SILVER" || t === "GOLD") return t;
  return null;
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

    const body = await request.json();
    const stripeSessionId = body?.stripeSessionId;

    if (!stripeSessionId) {
      return Response.json(
        { error: "stripeSessionId is required" },
        { status: 400 },
      );
    }

    const stripeRes = await fetch(
      `https://api.stripe.com/v1/checkout/sessions/${stripeSessionId}`,
      {
        method: "GET",
        headers: {
          Authorization: `Bearer ${process.env.STRIPE_SECRET_KEY}`,
        },
      },
    );

    if (!stripeRes.ok) {
      const errText = await stripeRes.text().catch(() => "");
      console.error("Stripe session fetch failed", errText);
      return Response.json(
        { error: "Could not verify Stripe session" },
        { status: 500 },
      );
    }

    const stripeJson = await stripeRes.json();

    const paymentStatus = stripeJson?.payment_status;
    const creditsFromMeta = Number(stripeJson?.metadata?.credits || 0);
    const packType = normalizePackType(stripeJson?.metadata?.packType);
    const creditsFromPack = packType
      ? Number(CREDIT_PACKS?.[packType]?.credits || 0)
      : 0;
    const credits = creditsFromMeta > 0 ? creditsFromMeta : creditsFromPack;

    const sessionUserIdRaw =
      stripeJson?.metadata?.userId || stripeJson?.metadata?.user_id;
    const sessionUserId = normalizeUserIdToUuid(sessionUserIdRaw);

    if (sessionUserId && sessionUserId !== userId) {
      return Response.json({ error: "Session user mismatch" }, { status: 403 });
    }

    if (paymentStatus !== "paid") {
      return Response.json({
        success: false,
        status: paymentStatus || "unknown",
      });
    }

    if (!credits || credits <= 0) {
      return Response.json(
        { error: "Invalid credits amount" },
        { status: 400 },
      );
    }

    // Idempotent finalization: rely on unique index (provider='stripe', provider_ref=stripeSessionId)
    // IMPORTANT: Anything sql.transaction() expects an array (or a function returning an array).
    // Use one atomic statement (CTEs) instead.
    const metaJson = JSON.stringify({
      packType,
      credits,
      // keep legacy field for older sessions
      packageId: stripeJson?.metadata?.package_id,
    });

    const out = await sql(
      `
        WITH inserted AS (
          INSERT INTO credit_transactions (
            user_id,
            transaction_type,
            credits_delta,
            provider,
            provider_ref,
            meta
          )
          VALUES ($1, 'purchase', $2, 'stripe', $3, $4::jsonb)
          ON CONFLICT DO NOTHING
          RETURNING id
        ),
        ensured_profile AS (
          INSERT INTO profiles (id, full_name)
          VALUES ($1, NULL)
          ON CONFLICT (id) DO NOTHING
        ),
        ensured_wallet AS (
          INSERT INTO credits_wallet (user_id, balance_credits)
          VALUES ($1, 0)
          ON CONFLICT (user_id) DO NOTHING
        ),
        updated_wallet AS (
          UPDATE credits_wallet
          SET balance_credits = balance_credits + $2
          WHERE user_id = $1
            AND EXISTS (SELECT 1 FROM inserted)
          RETURNING balance_credits
        )
        SELECT EXISTS(SELECT 1 FROM inserted) AS inserted;
      `,
      [userId, credits, stripeSessionId, metaJson],
    );

    const inserted = !!out?.[0]?.inserted;

    if (!inserted) {
      return Response.json({ success: true, alreadyProcessed: true });
    }

    return Response.json({ success: true, creditsAdded: credits });
  } catch (error) {
    console.error("POST /api/stripe/finalize error:", error);
    return Response.json({ error: "Internal Server Error" }, { status: 500 });
  }
}
