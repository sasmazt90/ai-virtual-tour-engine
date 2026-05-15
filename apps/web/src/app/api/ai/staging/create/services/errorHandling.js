import {
  getJobData,
  refundCreditsIfNeeded,
  markJobFailed,
} from "../utils/database";

export async function handleJobError({ jobId, error, debugPayload }) {
  console.error("processStagingJob error", error);

  try {
    const job = await getJobData(jobId);
    if (job) {
      const userId = job.user_id;
      const credits = Number(job.credits_reserved || 0);
      if (credits > 0) {
        await refundCreditsIfNeeded({ userId, credits, jobId });
      }
    }
  } catch (refundErr) {
    console.error("refundCredits error", refundErr);
  }

  const errorMessage = error?.message;

  // If this was VACANT, attach the per-image QA trace so the UI/debugger can show it.
  const attachDebug =
    debugPayload?.stagingType === "vacant" &&
    Array.isArray(debugPayload?.vacantQaResults) &&
    debugPayload.vacantQaResults.length > 0;

  await markJobFailed({
    jobId,
    errorMessage,
    resultPayload: attachDebug
      ? {
          errorType: "vacant_qa_failed",
          vacantQaResults: debugPayload.vacantQaResults,
        }
      : undefined,
  });
}
