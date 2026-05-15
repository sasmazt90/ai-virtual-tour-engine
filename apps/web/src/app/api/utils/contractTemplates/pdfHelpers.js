export function pdfEscapeLiteral(str) {
  return String(str || "")
    .replaceAll("\\", "\\\\")
    .replaceAll("(", "\\(")
    .replaceAll(")", "\\)")
    .replaceAll("\r", "")
    .replaceAll("\n", "\\n");
}

export function toUtf16BeHex(str) {
  const s = String(str || "");
  // Encode as UTF-16BE with BOM so Turkish characters render in PDF form fields.
  const buf = Buffer.from(s, "utf16le");
  // buf is LE; convert to BE
  const swapped = Buffer.alloc(buf.length);
  for (let i = 0; i < buf.length; i += 2) {
    swapped[i] = buf[i + 1];
    swapped[i + 1] = buf[i];
  }
  const bom = Buffer.from([0xfe, 0xff]);
  const full = Buffer.concat([bom, swapped]);
  return `<${full.toString("hex").toUpperCase()}>`;
}

export function safePdfName(str) {
  // PDF field names: keep simple ASCII
  return String(str || "")
    .replaceAll(/[^a-zA-Z0-9_\-\.]/g, "_")
    .slice(0, 64);
}
