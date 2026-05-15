export function formatMoney(value) {
  if (value === null || value === undefined) return "—";
  const num = Number(value);
  if (Number.isNaN(num)) return "—";

  // Use a fixed locale so SSR output matches the client.
  // User request: thousands ',' and decimals '.' (en-US style).
  try {
    return new Intl.NumberFormat("en-US", {
      maximumFractionDigits: 2,
    }).format(num);
  } catch {
    return String(num);
  }
}

export function parseNumberFromInput(input) {
  if (input === null || input === undefined) return null;
  const raw = String(input).trim();
  if (!raw) return null;

  // Accept digits, commas for thousands, dot for decimals.
  const cleaned = raw.replace(/,/g, "");
  const n = Number(cleaned);
  if (!Number.isFinite(n)) return null;
  return n;
}

export function formatNumberForInput(input) {
  const n = typeof input === "number" ? input : parseNumberFromInput(input);
  if (n === null || n === undefined) return "";
  if (!Number.isFinite(n)) return "";

  try {
    return new Intl.NumberFormat("en-US", {
      maximumFractionDigits: 2,
    }).format(n);
  } catch {
    return String(n);
  }
}

// Integer helpers (used for money-like fields where cents/decimals are not needed).
export function parseIntegerFromInput(input) {
  const n = parseNumberFromInput(input);
  if (n === null || n === undefined) return null;
  if (!Number.isFinite(n)) return null;
  return Math.trunc(n);
}

export function formatIntegerForInput(input) {
  const n = typeof input === "number" ? input : parseIntegerFromInput(input);
  if (n === null || n === undefined) return "";
  if (!Number.isFinite(n)) return "";

  try {
    return new Intl.NumberFormat("en-US", {
      maximumFractionDigits: 0,
    }).format(n);
  } catch {
    return String(n);
  }
}

export function titleCase(s) {
  if (!s) return "";
  return String(s)
    .split("_")
    .map((p) => p.charAt(0).toUpperCase() + p.slice(1))
    .join(" ");
}
