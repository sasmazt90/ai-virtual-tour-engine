async function uploadViaServer(file, { fallbackName, reportProgress }) {
  reportProgress(8);

  const response = await fetch("/api/upload/large", {
    method: "POST",
    headers: {
      "Content-Type": file.type || "application/octet-stream",
      "x-filename": encodeURIComponent(file.name || fallbackName || "upload"),
    },
    body: file,
  });

  reportProgress(95);

  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(body?.error || "Failed to upload large file.");
  }

  reportProgress(100);

  return {
    url: body.url,
    mimeType: body.mimeType || file.type || "application/octet-stream",
    sizeBytes: body.sizeBytes || file.size || null,
  };
}

export async function uploadLargeFile(file, { fallbackName = "upload", onProgress } = {}) {
  if (!file) throw new Error("Choose a file first.");

  const reportProgress = (value) => {
    if (typeof onProgress === "function") {
      onProgress(Math.max(0, Math.min(100, Math.round(value))));
    }
  };

  reportProgress(1);

  const signResponse = await fetch("/api/upload/large/sign", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      filename: file.name || fallbackName,
      mimeType: file.type || "application/octet-stream",
      sizeBytes: file.size || 0,
    }),
  });

  const signBody = await signResponse.json().catch(() => ({}));
  if (!signResponse.ok) {
    if (signResponse.status >= 500) {
      return uploadViaServer(file, { fallbackName, reportProgress });
    }
    throw new Error(signBody?.error || "Could not prepare upload.");
  }

  reportProgress(5);

  const { createClient } = await import("@supabase/supabase-js");
  const supabase = createClient(signBody.supabaseUrl, signBody.anonKey, {
    auth: {
      autoRefreshToken: false,
      persistSession: false,
    },
  });

  const { error } = await supabase.storage
    .from(signBody.bucket)
    .uploadToSignedUrl(signBody.path, signBody.token, file, {
      contentType: file.type || signBody.mimeType || "application/octet-stream",
      upsert: false,
    });

  if (error) {
    return uploadViaServer(file, { fallbackName, reportProgress });
  }

  reportProgress(100);

  return {
    url: signBody.publicUrl,
    mimeType: file.type || signBody.mimeType || "application/octet-stream",
    sizeBytes: file.size || signBody.sizeBytes || null,
  };
}
