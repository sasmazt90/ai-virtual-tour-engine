const STORAGE_URL_KEYS = new Set([
  "storage_path",
  "fileUrl",
  "url",
  "download_url",
  "data_url",
]);

function getSupabaseStorageConfig() {
  const url = process.env.SUPABASE_URL;
  const serviceRoleKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
  const bucket = process.env.SUPABASE_STORAGE_BUCKET || "uploads";

  if (!url || !serviceRoleKey) return null;

  return {
    url: url.replace(/\/+$/, ""),
    bucket,
    serviceRoleKey,
  };
}

export function collectStorageUrlsFromValue(value, out = new Set()) {
  if (!value) return out;

  if (typeof value === "string") {
    if (/^https?:\/\//i.test(value)) out.add(value.trim());
    return out;
  }

  if (Array.isArray(value)) {
    for (const item of value) collectStorageUrlsFromValue(item, out);
    return out;
  }

  if (typeof value === "object") {
    for (const [key, item] of Object.entries(value)) {
      if (STORAGE_URL_KEYS.has(key) && typeof item === "string") {
        collectStorageUrlsFromValue(item, out);
      } else if (item && typeof item === "object") {
        collectStorageUrlsFromValue(item, out);
      }
    }
  }

  return out;
}

export function objectPathFromSupabasePublicUrl(rawUrl) {
  const config = getSupabaseStorageConfig();
  if (!config || !rawUrl) return null;

  try {
    const parsed = new URL(String(rawUrl));
    const expectedOrigin = new URL(config.url).origin;
    if (parsed.origin !== expectedOrigin) return null;

    const marker = `/storage/v1/object/public/${encodeURIComponent(config.bucket)}/`;
    if (!parsed.pathname.startsWith(marker)) return null;

    return decodeURIComponent(parsed.pathname.slice(marker.length));
  } catch {
    return null;
  }
}

export async function deleteSupabaseStorageObjects(rawUrls) {
  const config = getSupabaseStorageConfig();
  const urls = Array.from(
    rawUrls instanceof Set ? rawUrls : collectStorageUrlsFromValue(rawUrls),
  );
  const objectPaths = Array.from(
    new Set(urls.map(objectPathFromSupabasePublicUrl).filter(Boolean)),
  );

  if (!config || objectPaths.length === 0) {
    return { attempted: 0, deleted: 0, failed: [] };
  }

  let deleted = 0;
  const failed = [];

  for (const objectPath of objectPaths) {
    const deleteUrl = `${config.url}/storage/v1/object/${encodeURIComponent(
      config.bucket,
    )}/${objectPath.split("/").map(encodeURIComponent).join("/")}`;

    try {
      const response = await fetch(deleteUrl, {
        method: "DELETE",
        headers: {
          apikey: config.serviceRoleKey,
          Authorization: `Bearer ${config.serviceRoleKey}`,
        },
      });

      const body = response.ok ? "" : await response.text().catch(() => "");
      const isAlreadyMissing =
        response.status === 404 ||
        /"statusCode"\s*:\s*"404"/i.test(body) ||
        /object not found|not_found/i.test(body);

      if (response.ok || isAlreadyMissing) {
        deleted += 1;
      } else {
        failed.push({
          objectPath,
          status: response.status,
          error: body || response.statusText,
        });
      }
    } catch (error) {
      failed.push({
        objectPath,
        error: error?.message || "Storage delete failed",
      });
    }
  }

  return { attempted: objectPaths.length, deleted, failed };
}
