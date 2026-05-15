export function copyToClipboard(text) {
  if (typeof navigator === "undefined" || !navigator.clipboard) {
    return false;
  }
  navigator.clipboard.writeText(text);
  return true;
}

export function buildOrigin() {
  return typeof window !== "undefined" ? window.location.origin : "";
}

export function isExpired(expiresAt) {
  if (!expiresAt) return false;
  const t = new Date(expiresAt).getTime();
  if (Number.isNaN(t)) return false;
  return t < Date.now();
}

export function getLastViewedAt(meta) {
  const access = meta && typeof meta === "object" ? meta.access : null;
  const arr = Array.isArray(access) ? access : [];
  let max = null;
  for (const a of arr) {
    const ts = a?.timestamp;
    if (!ts) continue;
    const d = new Date(ts);
    if (Number.isNaN(d.getTime())) continue;
    if (!max || d.getTime() > max.getTime()) max = d;
  }
  return max ? max.toISOString() : null;
}
