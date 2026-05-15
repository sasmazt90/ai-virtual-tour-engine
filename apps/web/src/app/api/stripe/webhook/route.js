import sql from "@/app/api/utils/sql";
import crypto from "crypto";
import { normalizeUserIdToUuid } from "../../utils/dbUser";
import { CREDIT_PACKS } from "../../utils/creditPacks";

function timingSafeEqualHex(a, b) {
  const aa = Buffer.from(String(a || ""), "utf8");
  const bb = Buffer.from(String(b || ""), "utf8");
  if (aa.length !== bb.length) return false;
  return crypto.timingSafeEqual(aa, bb);
}

function verifyStripeSignature({ payload, signatureHeader, webhookSecret }) {
  if (!signatureHeader || !webhookSecret) return false;

  const parts = String(signatureHeader)
    .split(",")
    .map((p) => p.trim())
    .filter(Boolean);

  const timestampPart = parts.find((p) => p.startsWith("t="));
  const v1Parts = parts.filter((p) => p.startsWith("v1="));
  if (!timestampPart || v1Parts.length === 0) return false;

  const timestamp = timestampPart.slice(2);
  const tsSec = Number(timestamp);
  if (!Number.isFinite(tsSec)) return false;

  // 5 min tolerance
  const nowSec = Math.floor(Date.now() / 1000);
  if (Math.abs(nowSec - tsSec) > 60 * 5) return false;

  const signedPayload = `${timestamp}.${payload}`;
  const expected = crypto
    .createHmac("sha256", webhookSecret)
    .update(signedPayload, "utf8")
    .digest("hex");

  for (const v of v1Parts) {
    const sig = v.slice(3);
    if (sig && timingSafeEqualHex(sig, expected)) {
      return true;
    }
  }

  return false;
}

function normalizePackType(raw) {
  const t = String(raw || "").toUpperCase();
  if (t === "BRONZE" || t === "SILVER" || t === "GOLD") return t;
  return null;
}

async function ensureProfileRow({ txn, userId }) {
  if (!userId) return;
  await txn(
    "INSERT INTO profiles (id, full_name) VALUES ($1, NULL) ON CONFLICT (id) DO NOTHING",
    [userId],
  );
}

async function grantCreditsFromStripeSession({ stripeSession }) {
  const stripeSessionId = stripeSession?.id;
  if (!stripeSessionId) return { ok: false, error: "Missing session id" };

  const meta = stripeSession?.metadata || {};
  const userId = normalizeUserIdToUuid(meta.userId || meta.user_id);
  if (!userId) return { ok: false, error: "Missing userId" };

  const packType = normalizePackType(meta.packType || meta.pack_type);
  const creditsFromPack = packType
    ? Number(CREDIT_PACKS?.[packType]?.credits || 0)
    : 0;
  const creditsFromMeta = Number(meta.credits || 0);
  const credits = creditsFromMeta > 0 ? creditsFromMeta : creditsFromPack;

  if (!credits || credits <= 0) return { ok: false, error: "Missing credits" };

  // IMPORTANT: Anything sql.transaction() expects an array (or a function returning an array).
  // Use one atomic statement (CTEs) instead.
  const metaJson = JSON.stringify({ packType, credits });

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
  return { ok: true, alreadyProcessed: !inserted };
}

export async function POST(request) {
  // Stripe webhooks must not require auth.
  try {
    const payload = await request.text();
    const signatureHeader = request.headers.get("stripe-signature");

    const ok = verifyStripeSignature({
      payload,
      signatureHeader,
      webhookSecret: process.env.STRIPE_WEBHOOK_KEY,
    });

    if (!ok) {
      return Response.json({ error: "Invalid signature" }, { status: 400 });
    }

    const event = JSON.parse(payload);

    if (event?.type === "checkout.session.completed") {
      const stripeSession = event?.data?.object;
      const res = await grantCreditsFromStripeSession({ stripeSession });
      if (!res.ok) {
        console.error("stripe webhook: could not grant credits", res.error);
      }
    }

    return Response.json({ received: true });
  } catch (error) {
    console.error("POST /api/stripe/webhook error", error);
    return Response.json({ error: "Webhook handler failed" }, { status: 500 });
  }
}
