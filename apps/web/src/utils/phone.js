export function normalizePhoneToE164(input) {
  // NOTE:
  // The app supports clients globally, so we normalize without assuming a default country.
  //
  // Expected best input: E.164 (e.g. +905xxxxxxxxx, +1xxxxxxxxxx, +44xxxxxxxxxx)
  // We try to be forgiving (strip spaces/dashes, convert 00 prefix to +).

  if (input === null || input === undefined) return null;

  const raw = String(input).trim();
  if (!raw) return null;

  const hasPlus = raw.startsWith("+");
  let digits = raw.replace(/\D/g, "");

  if (!digits) return null;

  // Convert 00 prefix to +
  if (digits.startsWith("00")) {
    digits = digits.slice(2);
  }

  // If the user provided a country code (either with + or 00), return E.164-ish.
  // E.164 max is 15 digits. Minimum varies; we accept 8 to avoid dropping real numbers.
  if (hasPlus || raw.startsWith("00")) {
    if (digits.length >= 8 && digits.length <= 15) {
      return `+${digits}`;
    }
  }

  // Otherwise: don't guess the country code. Keep original so we don't corrupt.
  return raw;
}

// Backwards-compatible alias.
// Some parts of the app still import a TR-specific helper name.
// We normalize globally (no default country guessing), but keep this export
// so publish/build doesn't break.
export function normalizePhoneToE164_TR(input) {
  return normalizePhoneToE164(input);
}
