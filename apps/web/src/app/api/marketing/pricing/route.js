const CACHE_KEY = "__marketing_pricing_cache_v1";

function getCache() {
  const v = globalThis[CACHE_KEY];
  if (!v || typeof v !== "object") return null;
  return v;
}

function setCache(value) {
  globalThis[CACHE_KEY] = value;
}

function normalizeStripePrice(p) {
  if (!p || typeof p !== "object") return null;

  const unitAmount =
    typeof p.unit_amount === "number" ? p.unit_amount : Number(p.unit_amount);
  const currency = typeof p.currency === "string" ? p.currency : null;

  const interval = p?.recurring?.interval || null;
  const productName = p?.product?.name || null;

  return {
    id: p.id || null,
    lookupKey: p.lookup_key || null,
    unitAmount: Number.isFinite(unitAmount) ? unitAmount : null,
    currency,
    interval,
    productName,
  };
}

export async function GET() {
  // Public endpoint. Never return secrets.
  const coinsIncluded = {
    starter: 100,
    pro: 250,
    agency: 500,
  };

  const baseResponse = {
    pricingPartial: false,
    fetchedAt: new Date().toISOString(),
    plans: {
      starter: { ...coinsIncluded, coins: coinsIncluded.starter, price: null },
      pro: { ...coinsIncluded, coins: coinsIncluded.pro, price: null },
      agency: { ...coinsIncluded, coins: coinsIncluded.agency, price: null },
    },
  };

  try {
    const url = new URL("https://api.stripe.com/v1/prices");
    url.searchParams.set("active", "true");
    url.searchParams.set("limit", "100");
    url.searchParams.append("lookup_keys[]", "starter");
    url.searchParams.append("lookup_keys[]", "pro");
    url.searchParams.append("lookup_keys[]", "agency");
    url.searchParams.append("expand[]", "data.product");

    const stripeRes = await fetch(url.toString(), {
      headers: {
        Authorization: `Bearer ${process.env.STRIPE_SECRET_KEY}`,
      },
    });

    if (!stripeRes.ok) {
      const cached = getCache();
      if (cached?.plans) {
        return Response.json(
          {
            ...cached,
            pricingPartial: true,
            fetchedAt: new Date().toISOString(),
          },
          { status: 200 },
        );
      }
      return Response.json(
        { ...baseResponse, pricingPartial: true },
        { status: 200 },
      );
    }

    const data = await stripeRes.json();
    const prices = Array.isArray(data?.data) ? data.data : [];

    const pickBest = (lookupKey) => {
      const candidates = prices.filter((p) => p?.lookup_key === lookupKey);
      if (candidates.length === 0) return null;

      // Prefer monthly recurring if present.
      const monthly = candidates.find(
        (p) => p?.recurring?.interval === "month",
      );
      return monthly || candidates[0];
    };

    const starterPrice = normalizeStripePrice(pickBest("starter"));
    const proPrice = normalizeStripePrice(pickBest("pro"));
    const agencyPrice = normalizeStripePrice(pickBest("agency"));

    const result = {
      pricingPartial: false,
      fetchedAt: new Date().toISOString(),
      plans: {
        starter: {
          coins: coinsIncluded.starter,
          price: starterPrice,
        },
        pro: {
          coins: coinsIncluded.pro,
          price: proPrice,
        },
        agency: {
          coins: coinsIncluded.agency,
          price: agencyPrice,
        },
      },
    };

    if (!starterPrice || !proPrice || !agencyPrice) {
      result.pricingPartial = true;
    }

    setCache(result);
    return Response.json(result);
  } catch (error) {
    console.error("GET /api/marketing/pricing error:", error);
    const cached = getCache();
    if (cached?.plans) {
      return Response.json(
        {
          ...cached,
          pricingPartial: true,
          fetchedAt: new Date().toISOString(),
        },
        { status: 200 },
      );
    }

    return Response.json(
      { ...baseResponse, pricingPartial: true },
      { status: 200 },
    );
  }
}
