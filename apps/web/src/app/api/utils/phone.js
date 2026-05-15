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

  if (digits.startsWith("00")) {
    digits = digits.slice(2);
  }

  if (hasPlus || raw.startsWith("00")) {
    if (digits.length >= 8 && digits.length <= 15) {
      return `+${digits}`;
    }
  }

  // Don't guess default country codes.
  return raw;
}
