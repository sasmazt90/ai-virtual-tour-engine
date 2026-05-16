import useUser from "@/utils/useUser";
import { Header } from "@/components/Header";
import { ArrowLeft } from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import {
  SHARE_LINK_DEFAULT_EXPIRY_DAYS,
  SHARE_LINK_MAX_EXPIRY_DAYS,
} from "@/utils/shareLinksConfig";
import {
  AI_STAGING_CREDIT_COST,
} from "@/app/api/utils/pricing";

function statusBadgeClass(status) {
  const s = String(status || "unknown");
  if (s === "configured") {
    return "bg-emerald-50 dark:bg-emerald-900/20 text-emerald-700 dark:text-emerald-200";
  }
  if (s === "missing") {
    return "bg-amber-50 dark:bg-amber-900/20 text-amber-800 dark:text-amber-200";
  }
  return "bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-200";
}

function statusBadgeText(status) {
  const s = String(status || "unknown");
  if (s === "configured") return "Configured";
  if (s === "missing") return "Missing";
  return "Unknown";
}

export default function ToolsConfigPage() {
  const { data: user, loading: userLoading } = useUser();

  // STEP 15A: read-only busy/idle badge (fetch once)
  const { data: busyData } = useQuery({
    queryKey: ["ai-busy", "tools", user?.id],
    queryFn: async () => {
      const res = await fetch("/api/ai/busy");
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body?.error || "Failed to load busy status");
      }
      const data = await res.json().catch(() => ({}));
      return {
        busy: !!data?.busy,
        queued: Number(data?.queued || 0),
        running: Number(data?.running || 0),
        partial: !!data?.partial,
      };
    },
    enabled: !!user?.id,
    // no polling here (fetch once)
    refetchOnWindowFocus: false,
    refetchOnReconnect: false,
  });

  const aiBusy = busyData?.busy === true;
  const aiBusyBadgeText = aiBusy ? "Processing" : "Ready";
  const aiBusyBadgeClass = aiBusy
    ? "bg-amber-50 dark:bg-amber-900/20 text-amber-800 dark:text-amber-200"
    : "bg-emerald-50 dark:bg-emerald-900/20 text-emerald-700 dark:text-emerald-200";

  const { data: aiStatus } = useQuery({
    queryKey: ["ai-status", user?.id],
    queryFn: async () => {
      const res = await fetch("/api/ai/status");
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body?.error || "Failed to load AI status");
      }
      return res.json();
    },
    enabled: !!user?.id,
  });

  const { data: shareStatus } = useQuery({
    queryKey: ["share-status", user?.id],
    queryFn: async () => {
      const res = await fetch("/api/share-links/status");
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body?.error || "Failed to load share status");
      }
      return res.json();
    },
    enabled: !!user?.id,
  });

  // STEP 17A/B/C: configuration + data integrity snapshot + static checklist
  const { data: toolsOverview } = useQuery({
    queryKey: ["tools-overview", user?.id],
    queryFn: async () => {
      const res = await fetch("/api/tools/overview");
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body?.error || "Failed to load tools overview");
      }
      return res.json();
    },
    enabled: !!user?.id,
    refetchOnWindowFocus: false,
    refetchOnReconnect: false,
  });

  if (userLoading) {
    return (
      <div className="min-h-screen bg-gray-50 dark:bg-[#1E1E1E]">
        <Header />
        <div className="pt-16 flex items-center justify-center min-h-screen">
          <p className="text-gray-600 dark:text-gray-400 font-jetbrains-mono">
            Loading…
          </p>
        </div>
      </div>
    );
  }

  if (!user) {
    if (typeof window !== "undefined") {
      window.location.href = "/account/signin";
    }
    return null;
  }

  const lastStagingStatus = aiStatus?.lastStagingJob?.job_status || "—";
  const lastTourStatus = aiStatus?.lastVirtualTourJob?.job_status || "—";

  const lastCompletedText = aiStatus?.lastCompletedAt
    ? new Date(aiStatus.lastCompletedAt).toLocaleString()
    : "—";

  const activeCount = Number(shareStatus?.activeCount || 0);
  const expiredCount = Number(shareStatus?.expiredCount || 0);
  const oldestActiveExpiryText = shareStatus?.oldestActiveExpiresAt
    ? new Date(shareStatus.oldestActiveExpiresAt).toLocaleString()
    : "—";

  const cfg = toolsOverview?.configurationStatus || {};
  const integrity = toolsOverview?.dataIntegrity || {};
  const overviewPartial = toolsOverview?.partial === true;

  const contractsOrphanCount =
    integrity?.contractsWithoutProperty === null ||
    integrity?.contractsWithoutProperty === undefined
      ? "—"
      : Number(integrity.contractsWithoutProperty);

  const stagingsOrphanCount =
    integrity?.stagingsWithoutProperty === null ||
    integrity?.stagingsWithoutProperty === undefined
      ? "—"
      : Number(integrity.stagingsWithoutProperty);

  const toursOrphanCount =
    integrity?.virtualToursWithoutProperty === null ||
    integrity?.virtualToursWithoutProperty === undefined
      ? "—"
      : Number(integrity.virtualToursWithoutProperty);

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-[#1E1E1E]">
      <Header />

      <div className="pt-16">
        <div className="max-w-4xl mx-auto px-4 sm:px-8 py-8 sm:py-12">
          <a
            href="/profile"
            className="inline-flex items-center gap-2 text-sm text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-gray-100 font-jetbrains-mono"
          >
            <ArrowLeft size={16} />
            Back to Profile
          </a>

          <div className="mt-6 mb-8">
            <div className="flex items-start justify-between gap-4">
              <div>
                <h1 className="text-3xl sm:text-4xl font-bold text-gray-900 dark:text-gray-100 mb-2 font-jetbrains-mono">
                  Internal Tools • Config
                </h1>
                <p className="text-gray-600 dark:text-gray-300 font-jetbrains-mono">
                  Read-only values for support/debugging.
                </p>

                {/* OPTIONAL: partial data indicator (subtle, non-blocking) */}
                {overviewPartial ? (
                  <div className="mt-2 text-xs text-gray-500 dark:text-gray-400 font-jetbrains-mono">
                    Some status information may be incomplete.
                  </div>
                ) : null}
              </div>

              {/* Top-right status pills (read-only) */}
              <div className="shrink-0 flex flex-col items-end gap-2">
                <div
                  className={`inline-flex items-center justify-center px-2 py-1 rounded-full text-xs font-jetbrains-mono leading-none whitespace-nowrap ${aiBusyBadgeClass}`}
                  title="Derived from /api/ai/busy"
                >
                  {aiBusyBadgeText}
                </div>

                {/* STEP 18B: release state lock (informational only) */}
                <div
                  className="inline-flex items-center justify-center px-2 py-1 rounded-full text-xs font-jetbrains-mono leading-none whitespace-nowrap bg-emerald-50 dark:bg-emerald-900/20 text-emerald-700 dark:text-emerald-200"
                  title="Informational only"
                >
                  Release status: Stable
                </div>
              </div>
            </div>
          </div>

          {/* NEW: quick link to E2E tests */}
          <div className="bg-white dark:bg-[#262626] rounded-xl shadow-lg dark:shadow-none dark:ring-1 dark:ring-gray-700 p-5 mb-6">
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
              <div>
                <div className="text-sm font-semibold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
                  AI E2E Tests
                </div>
                <div className="mt-1 text-xs text-gray-500 dark:text-gray-500 font-jetbrains-mono">
                  Run staging + virtual tour flows in a guided, repeatable way.
                </div>
              </div>
              <a
                href="/profile/tools/e2e"
                className="inline-flex items-center justify-center px-4 py-2 rounded-lg bg-gray-900 dark:bg-gray-100 text-white dark:text-gray-900 text-sm font-medium hover:bg-gray-800 dark:hover:bg-gray-200 font-jetbrains-mono"
              >
                Open E2E Tests
              </a>
            </div>
          </div>

          {/* STEP 13B/C: system status snapshot (read-only) */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-6">
            <div className="bg-white dark:bg-[#262626] rounded-xl shadow-lg dark:shadow-none dark:ring-1 dark:ring-gray-700 p-5">
              <div className="text-xs text-gray-500 dark:text-gray-400 font-jetbrains-mono">
                AI System Status
              </div>
              <div className="mt-3 space-y-2">
                <div className="flex items-center justify-between gap-3">
                  <div className="text-sm text-gray-600 dark:text-gray-300 font-jetbrains-mono">
                    Last staging job
                  </div>
                  <div className="text-sm text-gray-900 dark:text-gray-100 font-jetbrains-mono">
                    {lastStagingStatus}
                  </div>
                </div>
                <div className="flex items-center justify-between gap-3">
                  <div className="text-sm text-gray-600 dark:text-gray-300 font-jetbrains-mono">
                    Last virtual tour job
                  </div>
                  <div className="text-sm text-gray-900 dark:text-gray-100 font-jetbrains-mono">
                    {lastTourStatus}
                  </div>
                </div>
                <div className="flex items-center justify-between gap-3">
                  <div className="text-sm text-gray-600 dark:text-gray-300 font-jetbrains-mono">
                    Last completed at
                  </div>
                  <div className="text-sm text-gray-900 dark:text-gray-100 font-jetbrains-mono">
                    {lastCompletedText}
                  </div>
                </div>
              </div>
            </div>

            <div className="bg-white dark:bg-[#262626] rounded-xl shadow-lg dark:shadow-none dark:ring-1 dark:ring-gray-700 p-5">
              <div className="text-xs text-gray-500 dark:text-gray-400 font-jetbrains-mono">
                Share System Status
              </div>
              <div className="mt-3 space-y-2">
                <div className="flex items-center justify-between gap-3">
                  <div className="text-sm text-gray-600 dark:text-gray-300 font-jetbrains-mono">
                    Active links
                  </div>
                  <div className="text-sm text-gray-900 dark:text-gray-100 font-jetbrains-mono">
                    {activeCount}
                  </div>
                </div>
                <div className="flex items-center justify-between gap-3">
                  <div className="text-sm text-gray-600 dark:text-gray-300 font-jetbrains-mono">
                    Expired links
                  </div>
                  <div className="text-sm text-gray-900 dark:text-gray-100 font-jetbrains-mono">
                    {expiredCount}
                  </div>
                </div>
                <div className="flex items-center justify-between gap-3">
                  <div className="text-sm text-gray-600 dark:text-gray-300 font-jetbrains-mono">
                    Oldest active expiry
                  </div>
                  <div className="text-sm text-gray-900 dark:text-gray-100 font-jetbrains-mono">
                    {oldestActiveExpiryText}
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* STEP 17: release lockdown & handover readiness */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-6">
            <div className="bg-white dark:bg-[#262626] rounded-xl shadow-lg dark:shadow-none dark:ring-1 dark:ring-gray-700 p-5">
              <div className="text-xs text-gray-500 dark:text-gray-400 font-jetbrains-mono">
                Configuration Status
              </div>

              <div className="mt-3 space-y-3">
                <div className="flex items-center justify-between gap-3">
                  <div className="text-sm text-gray-600 dark:text-gray-300 font-jetbrains-mono">
                    Stripe
                  </div>
                  <div
                    className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-jetbrains-mono ${statusBadgeClass(cfg?.stripe)}`}
                  >
                    {statusBadgeText(cfg?.stripe)}
                  </div>
                </div>

                <div className="flex items-center justify-between gap-3">
                  <div className="text-sm text-gray-600 dark:text-gray-300 font-jetbrains-mono">
                    OpenAI
                  </div>
                  <div
                    className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-jetbrains-mono ${statusBadgeClass(cfg?.openai)}`}
                  >
                    {statusBadgeText(cfg?.openai)}
                  </div>
                </div>

                <div className="flex items-center justify-between gap-3">
                  <div className="text-sm text-gray-600 dark:text-gray-300 font-jetbrains-mono">
                    PDF Generation
                  </div>
                  <div
                    className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-jetbrains-mono ${statusBadgeClass(cfg?.pdfGeneration)}`}
                  >
                    {statusBadgeText(cfg?.pdfGeneration)}
                  </div>
                </div>
              </div>

              <div className="mt-3 text-xs text-gray-500 dark:text-gray-500 font-jetbrains-mono">
                Status is informational only. Secret values are never shown.
              </div>
            </div>

            <div className="bg-white dark:bg-[#262626] rounded-xl shadow-lg dark:shadow-none dark:ring-1 dark:ring-gray-700 p-5">
              <div className="text-xs text-gray-500 dark:text-gray-400 font-jetbrains-mono">
                Data Integrity
              </div>

              <div className="mt-3 space-y-2">
                <div className="flex items-center justify-between gap-3">
                  <div className="text-sm text-gray-600 dark:text-gray-300 font-jetbrains-mono">
                    Contracts without property
                  </div>
                  <div className="text-sm text-gray-900 dark:text-gray-100 font-jetbrains-mono">
                    {contractsOrphanCount}
                  </div>
                </div>
                <div className="flex items-center justify-between gap-3">
                  <div className="text-sm text-gray-600 dark:text-gray-300 font-jetbrains-mono">
                    Stagings without property
                  </div>
                  <div className="text-sm text-gray-900 dark:text-gray-100 font-jetbrains-mono">
                    {stagingsOrphanCount}
                  </div>
                </div>
                <div className="flex items-center justify-between gap-3">
                  <div className="text-sm text-gray-600 dark:text-gray-300 font-jetbrains-mono">
                    Virtual tours without property
                  </div>
                  <div className="text-sm text-gray-900 dark:text-gray-100 font-jetbrains-mono">
                    {toursOrphanCount}
                  </div>
                </div>
              </div>

              <div className="mt-3 text-xs text-gray-500 dark:text-gray-500 font-jetbrains-mono">
                Counts only. No actions.
              </div>
            </div>
          </div>

          {/* STEP 18A: internal handover summary (static, read-only) */}
          <div className="bg-white dark:bg-[#262626] rounded-xl shadow-lg dark:shadow-none dark:ring-1 dark:ring-gray-700 p-6 sm:p-8 mt-6">
            <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100 mb-3 font-jetbrains-mono">
              System Overview
            </h2>
            <p className="text-sm text-gray-700 dark:text-gray-200 font-jetbrains-mono">
              This app helps a real estate agent manage properties and client
              work, and generate AI-powered marketing assets and share links in
              a credit-based workflow.
            </p>
            <div className="mt-4 space-y-2 text-sm text-gray-700 dark:text-gray-200 font-jetbrains-mono">
              <div>
                • <span className="font-semibold">Properties & Clients</span>:
                Track listings, owners, and interested clients.
              </div>
              <div>
                •{" "}
                <span className="font-semibold">
                  AI Staging & Virtual Tours
                </span>
                : Create staging images and virtual tours; jobs are queued and
                tracked.
              </div>
              <div>
                • <span className="font-semibold">Contracts & PDFs</span>:
                Generate, store, and manage contract documents.
              </div>
              <div>
                • <span className="font-semibold">Share Links</span>: Create
                read-only share pages for clients with controlled expiry.
              </div>
              <div>
                • <span className="font-semibold">Credits & Payments</span>:
                Purchase credits via Stripe and spend credits on AI jobs.
              </div>
            </div>
            <div className="mt-4 text-xs text-gray-500 dark:text-gray-500 font-jetbrains-mono">
              This section is static and intended for handover / orientation.
            </div>
          </div>

          <div className="bg-white dark:bg-[#262626] rounded-xl shadow-lg dark:shadow-none dark:ring-1 dark:ring-gray-700 p-6 sm:p-8">
            <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100 mb-4 font-jetbrains-mono">
              Release Checklist
            </h2>
            <div className="space-y-2 text-sm text-gray-700 dark:text-gray-200 font-jetbrains-mono">
              <div>• Authentication enabled</div>
              <div>• Credits wallet active</div>
              <div>• AI jobs operational</div>
              <div>• Share links operational</div>
              <div>• Contract PDFs generating</div>
            </div>
            <div className="mt-4 text-xs text-gray-500 dark:text-gray-500 font-jetbrains-mono">
              This checklist is static and informational.
            </div>
          </div>

          <div className="bg-white dark:bg-[#262626] rounded-xl shadow-lg dark:shadow-none dark:ring-1 dark:ring-gray-700 p-6 sm:p-8">
            <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100 mb-4 font-jetbrains-mono">
              Share Links
            </h2>

            <div className="space-y-3">
              <div className="flex items-center justify-between gap-3">
                <div className="text-sm text-gray-600 dark:text-gray-300 font-jetbrains-mono">
                  Default expiry (days)
                </div>
                <div className="text-sm text-gray-900 dark:text-gray-100 font-jetbrains-mono">
                  {SHARE_LINK_DEFAULT_EXPIRY_DAYS}
                </div>
              </div>
              <div className="flex items-center justify-between gap-3">
                <div className="text-sm text-gray-600 dark:text-gray-300 font-jetbrains-mono">
                  Max expiry (days)
                </div>
                <div className="text-sm text-gray-900 dark:text-gray-100 font-jetbrains-mono">
                  {SHARE_LINK_MAX_EXPIRY_DAYS}
                </div>
              </div>
            </div>

            <div className="mt-8">
              <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100 mb-4 font-jetbrains-mono">
                Credit Costs
              </h2>
              <div className="space-y-3">
                <div className="flex items-center justify-between gap-3">
                  <div className="text-sm text-gray-600 dark:text-gray-300 font-jetbrains-mono">
                    AI Staging
                  </div>
                  <div className="text-sm text-gray-900 dark:text-gray-100 font-jetbrains-mono">
                    {AI_STAGING_CREDIT_COST}
                  </div>
                </div>
              </div>

              <div className="mt-4 text-xs text-gray-500 dark:text-gray-500 font-jetbrains-mono">
                This page is read-only. Editing happens in code/config.
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
