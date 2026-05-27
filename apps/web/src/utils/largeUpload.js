const TUS_CHUNK_SIZE = 6 * 1024 * 1024;
const S3_PROXY_PART_SIZE = 24 * 1024 * 1024;

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
    provider: body.provider || null,
    objectPath: body.objectPath || body.path || null,
    bucket: body.bucket || null,
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

async function uploadViaSignedPut(file, signBody, reportProgress) {
  if (!signBody?.signedUrl) {
    throw new Error("Could not prepare upload.");
  }

  reportProgress(5);

  await new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("PUT", signBody.signedUrl, true);
    xhr.upload.onprogress = (event) => {
      if (!event.lengthComputable) return;
      const percent = (event.loaded / event.total) * 95 + 5;
      reportProgress(percent);
    };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve();
        return;
      }
      reject(new Error(`Large file upload failed: ${xhr.status}`));
    };
    xhr.onerror = () => reject(new Error("Large file upload failed."));
    xhr.onabort = () => reject(new Error("Large file upload was cancelled."));
    xhr.send(file);
  });

  reportProgress(100);

  return {
    url: signBody.publicUrl || signBody.url,
    mimeType: file.type || signBody.mimeType || "application/octet-stream",
    sizeBytes: file.size || signBody.sizeBytes || null,
    provider: signBody.provider || "s3",
    objectPath: signBody.objectPath || signBody.path || null,
    bucket: signBody.bucket || null,
  };
}

async function postMultipartJson(action, signBody, body) {
  const response = await fetch(`/api/upload/large/multipart?action=${action}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-object-path": signBody.objectPath || signBody.path,
    },
    body: JSON.stringify(body || {}),
  });
  const responseBody = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(responseBody?.error || "Multipart upload failed.");
  }
  return responseBody;
}

function uploadMultipartPart(filePart, signBody, uploadId, partNumber) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/upload/large/multipart?action=part", true);
    xhr.setRequestHeader("x-object-path", signBody.objectPath || signBody.path);
    xhr.setRequestHeader("x-upload-id", uploadId);
    xhr.setRequestHeader("x-part-number", String(partNumber));
    xhr.setRequestHeader("Content-Type", "application/octet-stream");
    xhr.onload = () => {
      const body = (() => {
        try {
          return JSON.parse(xhr.responseText || "{}");
        } catch {
          return {};
        }
      })();
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(body);
        return;
      }
      reject(new Error(body?.error || `Multipart upload part ${partNumber} failed.`));
    };
    xhr.onerror = () => reject(new Error(`Multipart upload part ${partNumber} failed.`));
    xhr.onabort = () => reject(new Error(`Multipart upload part ${partNumber} was cancelled.`));
    xhr.send(filePart);
  });
}

async function uploadViaMultipartProxy(file, signBody, reportProgress) {
  const partSize = Math.max(
    5 * 1024 * 1024,
    Number(signBody.partSizeBytes || S3_PROXY_PART_SIZE),
  );
  const totalParts = Math.ceil(file.size / partSize);
  const init = await postMultipartJson("init", signBody);
  const uploadId = init.uploadId;
  const parts = [];

  if (!uploadId) {
    throw new Error("Could not prepare multipart upload.");
  }

  try {
    for (let index = 0; index < totalParts; index += 1) {
      const partNumber = index + 1;
      const start = index * partSize;
      const end = Math.min(file.size, start + partSize);
      const filePart = file.slice(start, end);
      let lastError = null;

      for (let attempt = 0; attempt < 3; attempt += 1) {
        try {
          const uploadedPart = await uploadMultipartPart(
            filePart,
            signBody,
            uploadId,
            partNumber,
          );
          parts.push(uploadedPart);
          lastError = null;
          break;
        } catch (error) {
          lastError = error;
          await new Promise((resolve) => setTimeout(resolve, 1000 * (attempt + 1)));
        }
      }

      if (lastError) throw lastError;

      reportProgress(5 + ((index + 1) / totalParts) * 90);
    }

    await postMultipartJson("complete", signBody, { uploadId, parts });
  } catch (error) {
    await postMultipartJson("abort", signBody, { uploadId }).catch(() => {});
    throw error;
  }

  reportProgress(100);

  return {
    url: signBody.publicUrl || signBody.url,
    mimeType: file.type || signBody.mimeType || "application/octet-stream",
    sizeBytes: file.size || signBody.sizeBytes || null,
    provider: signBody.provider || "s3",
    objectPath: signBody.objectPath || signBody.path || null,
    bucket: signBody.bucket || null,
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

  if (signBody?.uploadMethod === "server-proxy") {
    return uploadViaServer(file, { fallbackName, reportProgress });
  }

  if (signBody?.uploadMethod === "multipart-proxy") {
    return uploadViaMultipartProxy(file, signBody, reportProgress);
  }

  if (signBody?.uploadMethod === "signed-put") {
    try {
      return await uploadViaSignedPut(file, signBody, reportProgress);
    } catch (error) {
      console.warn("Signed large upload failed; retrying through server.", error);
      return uploadViaServer(file, { fallbackName, reportProgress });
    }
  }

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
