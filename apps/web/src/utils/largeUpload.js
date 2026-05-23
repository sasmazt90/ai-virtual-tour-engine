const TUS_CHUNK_SIZE = 6 * 1024 * 1024;

function resumableEndpointFor(supabaseUrl) {
  const parsed = new URL(supabaseUrl);
  const projectId = parsed.hostname.split(".")[0];

  if (projectId && parsed.hostname.endsWith(".supabase.co")) {
    return `https://${projectId}.storage.supabase.co/storage/v1/upload/resumable`;
  }

  return `${String(supabaseUrl || "").replace(/\/+$/, "")}/storage/v1/upload/resumable`;
}

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

async function uploadViaTus(file, signBody, reportProgress) {
  const tus = await import("tus-js-client");

  reportProgress(5);

  await new Promise((resolve, reject) => {
    const upload = new tus.Upload(file, {
      endpoint: resumableEndpointFor(signBody.supabaseUrl),
      retryDelays: [0, 3000, 5000, 10000, 20000],
      headers: {
        "x-signature": signBody.token,
        "x-upsert": "false",
      },
      metadata: {
        bucketName: signBody.bucket,
        objectName: signBody.path,
        contentType: file.type || signBody.mimeType || "application/octet-stream",
        cacheControl: "3600",
      },
      chunkSize: TUS_CHUNK_SIZE,
      uploadDataDuringCreation: true,
      removeFingerprintOnSuccess: true,
      onError: reject,
      onProgress: (bytesUploaded, bytesTotal) => {
        const total = Number(bytesTotal || file.size || 0);
        const uploaded = Number(bytesUploaded || 0);
        const percent = total > 0 ? (uploaded / total) * 95 + 5 : 5;
        reportProgress(percent);
      },
      onSuccess: resolve,
    });

    upload.findPreviousUploads().then((previousUploads) => {
      if (previousUploads.length) {
        upload.resumeFromPreviousUpload(previousUploads[0]);
      }
      upload.start();
    }).catch(reject);
  });

  reportProgress(100);

  return {
    url: signBody.publicUrl,
    mimeType: file.type || signBody.mimeType || "application/octet-stream",
    sizeBytes: file.size || signBody.sizeBytes || null,
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

  try {
    return await uploadViaTus(file, signBody, reportProgress);
  } catch (tusError) {
    if (file.size > TUS_CHUNK_SIZE) {
      throw new Error(
        tusError instanceof Error
          ? `Large file upload failed: ${tusError.message}`
          : "Large file upload failed.",
      );
    }
  }

  const { createClient } = await import("@supabase/supabase-js");
  if (!signBody.anonKey) {
    return uploadViaServer(file, { fallbackName, reportProgress });
  }

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
