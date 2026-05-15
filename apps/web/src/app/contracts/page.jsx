import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import useUser from "@/utils/useUser";
import { Header } from "../../components/Header";
import { Download, FileText, Plus, Search } from "lucide-react";

export default function ContractsPage() {
  const { data: user, loading: userLoading } = useUser();
  const [propertyId, setPropertyId] = useState("");
  const [clientId, setClientId] = useState("");
  const [search, setSearch] = useState("");

  const { data: properties = [] } = useQuery({
    queryKey: ["properties", user?.id, "contracts"],
    queryFn: async () => {
      const res = await fetch("/api/properties");
      if (!res.ok) throw new Error("Failed to load properties");
      return res.json();
    },
    enabled: !!user?.id,
  });

  const { data: clients = [] } = useQuery({
    queryKey: ["clients", user?.id, "contracts"],
    queryFn: async () => {
      const res = await fetch("/api/clients");
      if (!res.ok) throw new Error("Failed to load clients");
      return res.json();
    },
    enabled: !!user?.id,
  });

  const { data: contracts = [], isLoading } = useQuery({
    queryKey: ["contracts", user?.id, propertyId, clientId],
    queryFn: async () => {
      const params = new URLSearchParams();
      if (propertyId) params.set("propertyId", propertyId);
      if (clientId) params.set("clientId", clientId);
      const res = await fetch(`/api/contracts?${params.toString()}`);
      if (!res.ok) throw new Error("Failed to load contracts");
      return res.json();
    },
    enabled: !!user?.id,
  });

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q) return contracts;

    return contracts.filter((c) => {
      const t = (c.template_type || "").toLowerCase();
      const p = (c.property_title || "").toLowerCase();
      const cl = (c.client_name || "").toLowerCase();
      return t.includes(q) || p.includes(q) || cl.includes(q);
    });
  }, [contracts, search]);

  if (userLoading) {
    return (
      <div className="min-h-screen bg-gray-50 dark:bg-[#1E1E1E]">
        <Header />
        <div className="pt-16 flex items-center justify-center min-h-screen">
          <p className="text-gray-600 dark:text-gray-400 font-jetbrains-mono">
            Loading...
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

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-[#1E1E1E]">
      <Header />

      <div className="pt-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-8 py-8 sm:py-12">
          <div className="mb-8 flex items-start justify-between gap-4">
            <div>
              <h1 className="text-3xl sm:text-4xl font-bold text-gray-900 dark:text-gray-100 mb-2 font-jetbrains-mono">
                Contracts
              </h1>
              <p className="text-gray-600 dark:text-gray-300 font-jetbrains-mono">
                View and download generated PDFs.
              </p>
            </div>
            <a
              href="/contracts/new"
              className="inline-flex items-center justify-center gap-2 px-5 py-3 bg-gray-900 dark:bg-gray-100 text-white dark:text-gray-900 rounded-lg font-medium hover:bg-gray-800 dark:hover:bg-gray-200 transition-colors font-jetbrains-mono"
            >
              <Plus size={18} />
              New Contract
            </a>
          </div>

          <div className="mb-6 grid grid-cols-1 lg:grid-cols-3 gap-4">
            <div className="lg:col-span-1">
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 font-jetbrains-mono">
                Property
              </label>
              <select
                value={propertyId}
                onChange={(e) => setPropertyId(e.target.value)}
                className="mt-2 w-full px-4 py-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-[var(--brand)] font-jetbrains-mono"
              >
                <option value="">All properties</option>
                {properties.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.title}
                  </option>
                ))}
              </select>
            </div>

            <div className="lg:col-span-1">
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 font-jetbrains-mono">
                Client
              </label>
              <select
                value={clientId}
                onChange={(e) => setClientId(e.target.value)}
                className="mt-2 w-full px-4 py-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-[var(--brand)] font-jetbrains-mono"
              >
                <option value="">All clients</option>
                {clients.map((c) => (
                  <option key={c.id} value={c.id}>
                    {c.full_name}
                  </option>
                ))}
              </select>
            </div>

            <div className="lg:col-span-1">
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 font-jetbrains-mono">
                Search
              </label>
              <div className="mt-2 flex items-center gap-2 px-3 py-3 rounded-lg bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600">
                <Search size={16} className="text-gray-400" />
                <input
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  placeholder="template, property, client..."
                  className="w-full bg-transparent outline-none text-gray-900 dark:text-gray-100 placeholder-gray-500 dark:placeholder-gray-400 font-jetbrains-mono"
                />
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-[#262626] rounded-xl shadow-lg dark:shadow-none dark:ring-1 dark:ring-gray-700 overflow-hidden">
            {isLoading ? (
              <div className="p-8 text-gray-600 dark:text-gray-300 font-jetbrains-mono">
                Loading contracts...
              </div>
            ) : filtered.length === 0 ? (
              <div className="p-10 text-center">
                <FileText className="mx-auto mb-3 text-gray-400" size={44} />
                <p className="text-sm text-gray-600 dark:text-gray-400 font-jetbrains-mono">
                  No contracts.
                </p>
              </div>
            ) : (
              <div className="divide-y divide-gray-200 dark:divide-gray-700">
                {filtered.map((c) => {
                  const rawPdfState = c?.filled_fields?._system?.pdf || null;
                  const pdfStatus = rawPdfState?.status || null;
                  const pdfError = rawPdfState?.error || null;
                  const pdfStoragePath = rawPdfState?.storagePath || null;
                  const resolvedPdfUrl =
                    c.storage_path_pdf || pdfStoragePath || null;

                  const effectiveStatus =
                    pdfStatus || (resolvedPdfUrl ? "succeeded" : null);

                  const unavailableText =
                    effectiveStatus === "disabled"
                      ? "PDF disabled"
                      : "PDF unavailable";

                  const titleText = pdfError
                    ? `PDF: ${pdfError}`
                    : unavailableText;

                  const pending = !resolvedPdfUrl && !pdfStatus;

                  const downloadHref = resolvedPdfUrl
                    ? `/api/contracts/${c.id}/download?disposition=attachment`
                    : null;

                  return (
                    <div
                      key={c.id}
                      className="p-4 sm:p-5 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3"
                    >
                      <div className="min-w-0">
                        <a
                          href={`/contracts/${c.id}`}
                          className="text-sm sm:text-base font-semibold text-gray-900 dark:text-gray-100 hover:underline font-jetbrains-mono"
                        >
                          {c.template_type}
                        </a>
                        <div className="mt-1 text-xs text-gray-600 dark:text-gray-400 font-jetbrains-mono">
                          {c.property_title || "Property"} •{" "}
                          {c.client_name || "Client"} •{" "}
                          {new Date(c.created_at).toLocaleDateString()}
                        </div>
                      </div>

                      {downloadHref ? (
                        <a
                          href={downloadHref}
                          className="inline-flex items-center gap-2 px-3 py-2 rounded-lg bg-gray-900 dark:bg-gray-100 text-white dark:text-gray-900 text-sm font-medium hover:bg-gray-800 dark:hover:bg-gray-200 transition-colors font-jetbrains-mono"
                        >
                          <Download size={16} />
                          PDF
                        </a>
                      ) : pending ? (
                        <div className="text-xs text-gray-500 dark:text-gray-400 font-jetbrains-mono">
                          PDF pending
                        </div>
                      ) : (
                        <div
                          className="text-xs text-amber-700 dark:text-amber-300 font-jetbrains-mono"
                          title={titleText}
                        >
                          {unavailableText}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
