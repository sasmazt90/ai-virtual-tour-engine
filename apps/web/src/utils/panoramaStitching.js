function clamp(n, min, max) {
  return Math.min(max, Math.max(min, n));
}

export async function canvasToJpegBlob(canvas, quality = 0.86) {
  return new Promise((resolve, reject) => {
    try {
      canvas.toBlob(
        (blob) => {
          if (!blob) {
            reject(new Error("Could not encode panorama"));
            return;
          }
          resolve(blob);
        },
        "image/jpeg",
        quality,
      );
    } catch (e) {
      reject(e);
    }
  });
}

export async function loadScaledBitmap(file, maxLongEdgePx) {
  const bitmap = await createImageBitmap(file);
  const longEdge = Math.max(bitmap.width, bitmap.height);
  const scale = longEdge > maxLongEdgePx ? maxLongEdgePx / longEdge : 1;

  if (scale === 1) {
    return {
      bitmap,
      width: bitmap.width,
      height: bitmap.height,
    };
  }

  const w = Math.max(1, Math.round(bitmap.width * scale));
  const h = Math.max(1, Math.round(bitmap.height * scale));

  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(bitmap, 0, 0, w, h);

  // Free the original bitmap memory when possible.
  try {
    bitmap.close?.();
  } catch {
    // ignore
  }

  const scaled = await createImageBitmap(canvas);
  return { bitmap: scaled, width: w, height: h };
}

function toGrayArray(imageData) {
  const d = imageData.data;
  const out = new Float32Array(imageData.width * imageData.height);
  let j = 0;
  for (let i = 0; i < d.length; i += 4) {
    // cheap luma
    out[j] = d[i] * 0.299 + d[i + 1] * 0.587 + d[i + 2] * 0.114;
    j += 1;
  }
  return out;
}

function mse(a, b) {
  const n = Math.min(a.length, b.length);
  if (n <= 0) return Number.POSITIVE_INFINITY;
  let sum = 0;
  for (let i = 0; i < n; i += 1) {
    const diff = a[i] - b[i];
    sum += diff * diff;
  }
  return sum / n;
}

function estimateOverlapPx({ leftBitmap, rightBitmap, minPx, maxPx }) {
  // Compare the right edge of leftBitmap with the left edge of rightBitmap
  // across multiple candidate overlap widths.
  const outW = 140;
  const outH = 90;

  const cA = document.createElement("canvas");
  cA.width = outW;
  cA.height = outH;
  const aCtx = cA.getContext("2d", { willReadFrequently: true });

  const cB = document.createElement("canvas");
  cB.width = outW;
  cB.height = outH;
  const bCtx = cB.getContext("2d", { willReadFrequently: true });

  let best = { overlapPx: minPx, score: Number.POSITIVE_INFINITY };

  const step = Math.max(8, Math.round((maxPx - minPx) / 14));

  for (let overlap = minPx; overlap <= maxPx; overlap += step) {
    const lW = leftBitmap.width;
    const rW = rightBitmap.width;

    const srcOverlap = clamp(overlap, 8, Math.min(lW - 1, rW - 1));

    // Draw right strip from left
    aCtx.clearRect(0, 0, outW, outH);
    aCtx.drawImage(
      leftBitmap,
      lW - srcOverlap,
      0,
      srcOverlap,
      leftBitmap.height,
      0,
      0,
      outW,
      outH,
    );

    // Draw left strip from right
    bCtx.clearRect(0, 0, outW, outH);
    bCtx.drawImage(
      rightBitmap,
      0,
      0,
      srcOverlap,
      rightBitmap.height,
      0,
      0,
      outW,
      outH,
    );

    const gA = toGrayArray(aCtx.getImageData(0, 0, outW, outH));
    const gB = toGrayArray(bCtx.getImageData(0, 0, outW, outH));

    const score = mse(gA, gB);
    if (score < best.score) {
      best = { overlapPx: srcOverlap, score };
    }
  }

  return best.overlapPx;
}

function stitchTwo({ baseCanvas, baseBitmap, nextBitmap, overlapPx }) {
  const outH = Math.max(baseCanvas.height, nextBitmap.height);
  const outW = baseCanvas.width + nextBitmap.width - overlapPx;

  const out = document.createElement("canvas");
  out.width = Math.max(1, outW);
  out.height = Math.max(1, outH);

  const ctx = out.getContext("2d");

  const baseY = Math.round((outH - baseCanvas.height) / 2);
  const nextY = Math.round((outH - nextBitmap.height) / 2);

  // draw base
  ctx.drawImage(baseCanvas, 0, baseY);

  // blend overlap: fade in next across overlap region
  const overlap = clamp(
    overlapPx,
    0,
    Math.min(baseCanvas.width, nextBitmap.width),
  );
  const overlapX = baseCanvas.width - overlap;

  if (overlap > 0) {
    const slice = document.createElement("canvas");
    slice.width = overlap;
    slice.height = nextBitmap.height;
    const sctx = slice.getContext("2d");

    // 1) draw overlap slice from next
    sctx.drawImage(
      nextBitmap,
      0,
      0,
      overlap,
      nextBitmap.height,
      0,
      0,
      overlap,
      nextBitmap.height,
    );

    // 2) apply alpha gradient mask (0 -> 1)
    const mask = sctx.createLinearGradient(0, 0, overlap, 0);
    mask.addColorStop(0, "rgba(0,0,0,0)");
    mask.addColorStop(1, "rgba(0,0,0,1)");
    sctx.globalCompositeOperation = "destination-in";
    sctx.fillStyle = mask;
    sctx.fillRect(0, 0, overlap, nextBitmap.height);
    sctx.globalCompositeOperation = "source-over";

    // 3) draw blended overlap onto output
    ctx.drawImage(slice, overlapX, nextY);
  }

  // draw the rest of next (non-overlap)
  const remainingW = nextBitmap.width - overlap;
  if (remainingW > 0) {
    ctx.drawImage(
      nextBitmap,
      overlap,
      0,
      remainingW,
      nextBitmap.height,
      overlapX + overlap,
      nextY,
      remainingW,
      nextBitmap.height,
    );
  }

  return out;
}

export async function stitchBitmaps({ bitmaps, overlapMode }) {
  if (!Array.isArray(bitmaps) || bitmaps.length < 2) {
    throw new Error("Pick at least 2 photos to stitch");
  }

  // Start with first bitmap
  let base = document.createElement("canvas");
  base.width = bitmaps[0].bitmap.width;
  base.height = bitmaps[0].bitmap.height;
  base.getContext("2d").drawImage(bitmaps[0].bitmap, 0, 0);

  for (let i = 1; i < bitmaps.length; i += 1) {
    const left = base;
    const right = bitmaps[i].bitmap;

    const minDim = Math.min(left.width, right.width);
    const minPx = Math.max(40, Math.round(minDim * 0.12));
    const maxPx = Math.max(minPx + 10, Math.round(minDim * 0.48));

    let overlapPx;
    if (overlapMode?.mode === "manual") {
      const pct = clamp(Number(overlapMode?.manualPct || 30), 5, 80);
      overlapPx = Math.round((Math.min(left.width, right.width) * pct) / 100);
    } else {
      overlapPx = estimateOverlapPx({
        leftBitmap: left,
        rightBitmap: right,
        minPx,
        maxPx,
      });
    }

    const nextBase = stitchTwo({
      baseCanvas: left,
      baseBitmap: left,
      nextBitmap: right,
      overlapPx,
    });

    base = nextBase;
  }

  return base;
}
