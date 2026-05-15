export function clamp(n, min, max) {
  return Math.min(max, Math.max(min, n));
}

export function fillMissingCircular(urls) {
  const n = urls.length;
  if (!n) return urls;

  const hasAny = urls.some(Boolean);
  if (!hasAny) return urls;

  const out = [...urls];
  for (let i = 0; i < n; i++) {
    if (out[i]) continue;

    for (let d = 1; d < n; d++) {
      const left = (i - d + n) % n;
      const right = (i + d) % n;
      if (out[left]) {
        out[i] = out[left];
        break;
      }
      if (out[right]) {
        out[i] = out[right];
        break;
      }
    }
  }

  return out;
}

export function normalizeDeg(deg) {
  const v = ((deg % 360) + 360) % 360;
  return v;
}
