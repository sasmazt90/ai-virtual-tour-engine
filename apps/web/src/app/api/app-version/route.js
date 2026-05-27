export async function GET() {
  const version =
    process.env.RENDER_GIT_COMMIT ||
    process.env.COMMIT_SHA ||
    process.env.SOURCE_VERSION ||
    "local";

  return Response.json(
    {
      version,
      builtAt: process.env.RENDER_GIT_COMMIT ? null : new Date().toISOString(),
    },
    {
      headers: {
        "Cache-Control": "no-store, no-cache, must-revalidate",
      },
    },
  );
}
