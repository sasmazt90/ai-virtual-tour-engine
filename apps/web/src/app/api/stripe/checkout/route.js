import { auth } from "@/auth";
import { getDbUserIdFromSession } from "@/app/api/utils/dbUser";
import {
  CREDIT_PACKS,
  getStripePriceIdForPackType,
} from "@/app/api/utils/creditPacks";

function legacyPackageIdToPackType(packageId) {
  if (packageId === "credits_100") return "BRONZE";
  if (packageId === "credits_300") return "SILVER";
  if (packageId === "credits_800") return "GOLD";
  return null;
}

export async function POST(request) {
  try {
    const session = await auth();
    if (!session || !session.user?.id || !session.user?.email) {
      return Response.json({ error: "Unauthorized" }, { status: 401 });
    }

    const dbUserId = await getDbUserIdFromSession(session);
    if (!dbUserId) {
      return Response.json({ error: "Unauthorized" }, { status: 401 });
    }

    const body = await request.json();
    const packTypeRaw = body?.packType;
    const legacyPackageId = body?.packageId;
    const inferredPackType =
      packTypeRaw || legacyPackageIdToPackType(legacyPackageId);
    const packType =
      typeof inferredPackType === "string"
        ? inferredPackType.toUpperCase()
        : null;
    const pack = packType ? CREDIT_PACKS[packType] : null;

    if (!packType || !pack) {
      return Response.json(
        { error: "Invalid packType. Expected BRONZE, SILVER, or GOLD." },
        { status: 400 },
      );
    }

    const origin = new URL(request.url).origin;

    // IMPORTANT: Use mapped LIVE Stripe Price IDs (EUR) as the single source of truth.
    // Never use price_data / unit_amount here.
    const priceId = getStripePriceIdForPackType(packType);
    if (!priceId) {
      console.error("Stripe: missing priceId for pack", packType);
      return Response.json(
        { error: "Could not start checkout. Stripe price not configured." },
        { status: 500 },
      );
    }

    // Stripe REST API call (no SDK). Uses platform-provided STRIPE_SECRET_KEY.
    const form = new URLSearchParams();
    form.set("mode", "payment");
    form.set("payment_method_types[0]", "card");

    form.set("line_items[0][quantity]", "1");
    form.set("line_items[0][price]", priceId);

    form.set(
      "success_url",
      `${origin}/credits?stripeSessionId={CHECKOUT_SESSION_ID}`,
    );
    form.set("cancel_url", `${origin}/credits`);

    form.set("metadata[userId]", dbUserId);
    form.set("metadata[packType]", packType);
    form.set("metadata[credits]", String(pack.credits));

    const stripeRes = await fetch(
      "https://api.stripe.com/v1/checkout/sessions",
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${process.env.STRIPE_SECRET_KEY}`,
          "Content-Type": "application/x-www-form-urlencoded",
        },
        body: form.toString(),
      },
    );

    if (!stripeRes.ok) {
      const errText = await stripeRes.text().catch(() => "");
      console.error("Stripe checkout create failed", errText);
      return Response.json(
        { error: "Could not create checkout session" },
        { status: 500 },
      );
    }

    const stripeJson = await stripeRes.json();
    return Response.json({ checkoutUrl: stripeJson.url });
  } catch (error) {
    console.error("POST /api/stripe/checkout error:", error);
    return Response.json({ error: "Internal Server Error" }, { status: 500 });
  }
}
