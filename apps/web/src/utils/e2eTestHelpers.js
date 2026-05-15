export function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export async function pollJob({ jobId, timeoutMs }) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    const res = await fetch(`/api/ai/jobs/${jobId}`);
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body?.error || `Could not poll job ${jobId}`);
    }

    const data = await res.json();
    const st = data?.status;

    if (st === "succeeded" || st === "failed") {
      return data;
    }

    await sleep(2000);
  }

  throw new Error("Timed out while waiting for job to complete.");
}
