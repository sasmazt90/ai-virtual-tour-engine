// Single source of truth for credit packs.
// IMPORTANT:
// - Credits are granted based on packType -> credits mapping.
// - Stripe checkout MUST use LIVE Price IDs from Stripe Product Catalog (EUR).

const CREDIT_PACKS = {
  BRONZE: {
    credits: 100,
    // LIVE EUR price id from Stripe Product Catalog
    priceId: "price_1Sio10GwAcvquXHEDquzjZzj",
  },
  SILVER: {
    credits: 300,
    // LIVE EUR price id from Stripe Product Catalog
    priceId: "price_1Sio2tGwAcvquXHEPDNADcp0",
  },
  GOLD: {
    credits: 800,
    // LIVE EUR price id from Stripe Product Catalog
    priceId: "price_1Sio4nGwAcvquXHEbKusf0p3",
  },
};

function normalizePackType(packType) {
  const t = String(packType || "").toUpperCase();
  if (t === "BRONZE" || t === "SILVER" || t === "GOLD") return t;
  return null;
}

export function getCreditsForPackType(packType) {
  const t = normalizePackType(packType);
  const pack = t ? CREDIT_PACKS[t] : null;
  return pack ? Number(pack.credits || 0) : 0;
}

export function getStripePriceIdForPackType(packType) {
  const t = normalizePackType(packType);
  const pack = t ? CREDIT_PACKS[t] : null;
  return pack?.priceId ? String(pack.priceId) : null;
}

export function isValidPackType(packType) {
  return !!normalizePackType(packType);
}

export { CREDIT_PACKS };
