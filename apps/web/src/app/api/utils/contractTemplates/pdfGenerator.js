import { buildFillableContractFieldLayout } from "./pdfFieldLayout";
import { pdfEscapeLiteral, toUtf16BeHex, safePdfName } from "./pdfHelpers";

export function generateFillablePdfBytes({
  templateType,
  fields,
  agent,
  contractMeta,
}) {
  const { header, footer, fieldsLayout, contentBlocks } =
    buildFillableContractFieldLayout({
      templateType,
      fields,
      agent,
      contractMeta,
    });

  const objects = [];
  const offsets = [0]; // 1-based object index; offsets[0] unused

  const pushObject = (body) => {
    objects.push(body);
    return objects.length;
  };

  const buildObj = (n, body) => `${n} 0 obj\n${body}\nendobj\n`;

  // Predeclare numbers
  const catalogNum = 1;
  const pagesNum = 2;
  const pageNum = 3;
  const fontNum = 4;
  const contentsNum = 5;
  const acroFormNum = 6;

  // Fields start at 7
  const fieldStartNum = 7;

  // Content stream: layout + labels
  const lines = [];

  // Title
  lines.push("BT");
  lines.push("/Helv 18 Tf");
  lines.push("50 805 Td");
  lines.push(`(${pdfEscapeLiteral(header.title)}) Tj`);
  lines.push("ET");

  if (header.companyName) {
    lines.push("BT");
    lines.push("/Helv 11 Tf");
    lines.push("50 788 Td");
    lines.push(`(${pdfEscapeLiteral(header.companyName)}) Tj`);
    lines.push("ET");
  }

  // Additional static content blocks (Turkish body text, section headings, signature lines, etc.)
  const blocks = Array.isArray(contentBlocks) ? contentBlocks : [];
  for (const b of blocks) {
    if (!b || typeof b !== "object") continue;

    if (b.type === "text") {
      const x = Number(b.x || 0);
      const y = Number(b.y || 0);
      const size = Number(b.size || 10);
      const text = typeof b.text === "string" ? b.text : "";

      lines.push("BT");
      lines.push(`/Helv ${size} Tf`);
      lines.push(`${x} ${y} Td`);
      lines.push(`(${pdfEscapeLiteral(text)}) Tj`);
      lines.push("ET");
    }

    if (b.type === "line") {
      const x1 = Number(b.x1 || 0);
      const y1 = Number(b.y1 || 0);
      const x2 = Number(b.x2 || 0);
      const y2 = Number(b.y2 || 0);
      const w = Number(b.width || 1);

      // Draw a simple line
      lines.push(`${w} w`);
      lines.push("0 0 0 RG");
      lines.push(`${x1} ${y1} m`);
      lines.push(`${x2} ${y2} l`);
      lines.push("S");
    }
  }

  // Field labels
  for (const f of fieldsLayout) {
    if (f.kind === "text") {
      const labelX = f.rect[0];
      const labelY = f.rect[3] + 6;
      lines.push("BT");
      lines.push("/Helv 9 Tf");
      lines.push(`${labelX} ${labelY} Td`);
      lines.push(`(${pdfEscapeLiteral(f.label)}) Tj`);
      lines.push("ET");
    } else if (f.kind === "checkbox") {
      const labelX = f.rect[2] + 6;
      const labelY = f.rect[1] + 2;
      lines.push("BT");
      lines.push("/Helv 9 Tf");
      lines.push(`${labelX} ${labelY} Td`);
      lines.push(`(${pdfEscapeLiteral(f.label)}) Tj`);
      lines.push("ET");
    }
  }

  // Footer (localized if provided)
  const footerLines = Array.isArray(footer?.lines) ? footer.lines : null;
  if (footerLines && footerLines.length > 0) {
    let fy = 60;
    for (const text of footerLines.slice(0, 4)) {
      lines.push("BT");
      lines.push("/Helv 9 Tf");
      lines.push(`50 ${fy} Td`);
      lines.push(`(${pdfEscapeLiteral(String(text || ""))}) Tj`);
      lines.push("ET");
      fy -= 12;
    }
  } else {
    // Legacy fallback
    lines.push("BT");
    lines.push("/Helv 9 Tf");
    lines.push("50 60 Td");
    lines.push(`(Agent: ${pdfEscapeLiteral(footer.agentName || "—")}) Tj`);
    lines.push("ET");

    if (footer.companyName) {
      lines.push("BT");
      lines.push("/Helv 9 Tf");
      lines.push("50 48 Td");
      lines.push(`(Company: ${pdfEscapeLiteral(footer.companyName)}) Tj`);
      lines.push("ET");
    }

    lines.push("BT");
    lines.push("/Helv 9 Tf");
    lines.push("50 36 Td");
    lines.push(`(Date: ${pdfEscapeLiteral(footer.genDate)}) Tj`);
    lines.push("ET");

    lines.push("BT");
    lines.push("/Helv 9 Tf");
    lines.push("50 24 Td");
    lines.push(
      `(Version: ${pdfEscapeLiteral(String(footer.version || 1))}) Tj`,
    );
    lines.push("ET");
  }

  const contentStream = lines.join("\n") + "\n";
  const contentObjBody = `<< /Length ${Buffer.byteLength(contentStream, "utf8")} >>\nstream\n${contentStream}endstream`;

  // Build field objects
  const fieldObjNums = [];
  const annots = [];

  let nextFieldNum = fieldStartNum;
  for (const f of fieldsLayout) {
    const objNum = nextFieldNum;
    nextFieldNum += 1;

    const rect = `[${f.rect.map((n) => Number(n).toFixed(2)).join(" ")}]`;

    if (f.kind === "text") {
      const fieldName = safePdfName(f.name);
      const fieldValue = toUtf16BeHex(f.value || "");

      const body = `<<
/FT /Tx
/T (${pdfEscapeLiteral(fieldName)})
/V ${fieldValue}
/DA (/Helv 10 Tf 0 g)
/Q 0
/Type /Annot
/Subtype /Widget
/Rect ${rect}
/P ${pageNum} 0 R
/F 4
/Border [0 0 1]
>>`;

      fieldObjNums.push(`${objNum} 0 R`);
      annots.push(`${objNum} 0 R`);
      pushObject(buildObj(objNum, body));
    } else if (f.kind === "checkbox") {
      const fieldName = safePdfName(f.name);
      const onName = "/Yes";
      const v = f.value ? onName : "/Off";

      const body = `<<
/FT /Btn
/T (${pdfEscapeLiteral(fieldName)})
/V ${v}
/AS ${v}
/DA (/Helv 10 Tf 0 g)
/Type /Annot
/Subtype /Widget
/Rect ${rect}
/P ${pageNum} 0 R
/F 4
/Border [0 0 1]
>>`;

      fieldObjNums.push(`${objNum} 0 R`);
      annots.push(`${objNum} 0 R`);
      pushObject(buildObj(objNum, body));
    }
  }

  // Objects 1..6 are fixed; we have already pushed field objects at their final numbers.
  // Ensure objects array has placeholders up to the max object index.
  const maxObjNum = nextFieldNum - 1;
  const objBodies = new Array(maxObjNum + 1).fill(null);

  // Place field objects we created
  for (const objText of objects) {
    const match = objText.match(/^(\d+) 0 obj/);
    if (match) {
      const n = Number(match[1]);
      objBodies[n] = objText;
    }
  }

  // Fill fixed objects
  objBodies[catalogNum] = buildObj(
    catalogNum,
    `<< /Type /Catalog /Pages ${pagesNum} 0 R /AcroForm ${acroFormNum} 0 R >>`,
  );

  objBodies[pagesNum] = buildObj(
    pagesNum,
    `<< /Type /Pages /Kids [${pageNum} 0 R] /Count 1 >>`,
  );

  objBodies[pageNum] = buildObj(
    pageNum,
    `<<
/Type /Page
/Parent ${pagesNum} 0 R
/MediaBox [0 0 595 842]
/Resources << /Font << /Helv ${fontNum} 0 R >> >>
/Contents ${contentsNum} 0 R
/Annots [${annots.join(" ")}]
>>`,
  );

  objBodies[fontNum] = buildObj(
    fontNum,
    `<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Name /Helv >>`,
  );

  objBodies[contentsNum] = buildObj(contentsNum, contentObjBody);

  objBodies[acroFormNum] = buildObj(
    acroFormNum,
    `<<
/NeedAppearances true
/DA (/Helv 10 Tf 0 g)
/DR << /Font << /Helv ${fontNum} 0 R >> >>
/Fields [${fieldObjNums.join(" ")}]
>>`,
  );

  // Write file
  let out = "%PDF-1.7\n%\xE2\xE3\xCF\xD3\n";

  for (let i = 1; i <= maxObjNum; i += 1) {
    const body = objBodies[i];
    if (!body) {
      // Shouldn't happen, but keep xref aligned
      objBodies[i] = buildObj(i, "<<>>");
    }
    offsets[i] = Buffer.byteLength(out, "binary");
    out += objBodies[i];
  }

  const xrefOffset = Buffer.byteLength(out, "binary");
  out += "xref\n";
  out += `0 ${maxObjNum + 1}\n`;
  out += "0000000000 65535 f \n";
  for (let i = 1; i <= maxObjNum; i += 1) {
    const off = String(offsets[i] || 0).padStart(10, "0");
    out += `${off} 00000 n \n`;
  }

  out += "trailer\n";
  out += `<< /Size ${maxObjNum + 1} /Root ${catalogNum} 0 R >>\n`;
  out += "startxref\n";
  out += `${xrefOffset}\n`;
  out += "%%EOF\n";

  return Buffer.from(out, "binary");
}
