import { Header } from "../../components/Header";
import { useEffect, useMemo, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import useUser from "@/utils/useUser";
import { Coins, ShoppingCart } from "lucide-react";
import { StatusBanner } from "@/components/StatusBanner";

const CREDIT_PACKAGES = [
  {
    packType: "BRONZE",
    credits: 100,
    label: "Bronze Pack",
    priceEur: "€49.99",
  },
  {
    packType: "SILVER",
    credits: 300,
    label: "Silver Pack",
    priceEur: "€99.99",
  },
  {
    packType: "GOLD",
    credits: 800,
    label: "Gold Pack",
    priceEur: "€174.99",
  },
];

export default function CreditsPage() {
  const queryClient = useQueryClient();
  const { data: user, loading: userLoading } = useUser();
  const [finalizeMessage, setFinalizeMessage] = useState(null);

  const {
    data: creditsData,
    error: creditsError,
    isLoading: creditsLoading,
  } = useQuery({
    queryKey: ["credits", user?.id],
    queryFn: async () => {
      const res = await fetch("/api/credits");
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body?.error || "Failed to fetch credits");
      }
      return res.json();
    },
    enabled: !!user?.id,
  });

  const { data: transactions, error: transactionsError } = useQuery({
    queryKey: ["credit-transactions", user?.id],
    queryFn: async () => {
      const res = await fetch("/api/credits/transactions");
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body?.error || "Failed to fetch transactions");
      }
      return res.json();
    },
    enabled: !!user?.id,
  });

  const usageTotals = useMemo(() => {
    const arr = Array.isArray(transactions) ? transactions : [];

    const refundedJobIds = new Set();
    for (const t of arr) {
      if (t?.transaction_type !== "refund") continue;
      const jobId = t?.meta?.jobId;
      if (jobId) refundedJobIds.add(String(jobId));
    }

    let stagingSpent = 0;
    let tourSpent = 0;

    for (const t of arr) {
      if (t?.transaction_type !== "spend") continue;
      if (t?.meta?.kind !== "reserve") continue;

      const delta = Number(t?.credits_delta || 0);
      if (!Number.isFinite(delta) || delta >= 0) continue;

      const jobId = t?.meta?.jobId ? String(t.meta.jobId) : null;
      if (!jobId) continue;
      if (refundedJobIds.has(jobId)) continue;

      if (t?.ai_job_status !== "succeeded") continue;

      const spent = -delta;
      const jobType = t?.meta?.jobType;

      if (jobType === "staging") {
        stagingSpent += spent;
      } else if (jobType === "virtual_tour") {
        tourSpent += spent;
      }
    }

    return { stagingSpent, tourSpent };
  }, [transactions]);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }

    const params = new URLSearchParams(window.location.search);
    const stripeSessionId = params.get("stripeSessionId");

    if (!stripeSessionId) {
      return;
    }

    const finalize = async () => {
      try {
        setFinalizeMessage("Finalizing your purchase...");
        const res = await fetch("/api/stripe/finalize", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ stripeSessionId }),
        });
        if (!res.ok) {
          throw new Error("Could not finalize purchase");
        }
        const data = await res.json();
        if (data.success) {
          setFinalizeMessage("Credits added successfully.");
          queryClient.invalidateQueries({ queryKey: ["credits", user?.id] });
          queryClient.invalidateQueries({
            queryKey: ["credit-transactions", user?.id],
          });
        } else {
          setFinalizeMessage("Payment not completed yet.");
        }
      } catch (e) {
        console.error(e);
        setFinalizeMessage("Could not finalize purchase.");
      } finally {
        try {
          const url = new URL(window.location.href);
          url.searchParams.delete("stripeSessionId");
          window.history.replaceState({}, "", url.toString());
        } catch {
          // no-op
        }
      }
    };

    finalize();
  }, [queryClient, user?.id]);

  const handlePurchase = async (packType) => {
    try {
      const res = await fetch("/api/stripe/checkout", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ packType }),
      });

      if (!res.ok) {
        throw new Error("Failed to start checkout");
      }

      const { checkoutUrl } = await res.json();
      if (checkoutUrl && typeof window !== "undefined") {
        window.open(checkoutUrl, "_blank", "popup");
      }
    } catch (error) {
      console.error("Failed to initiate checkout:", error);
      setFinalizeMessage("Could not start checkout.");
    }
  };

  if (userLoading) {
    return (
      <div className="min-h-screen ui-surface">
        <Header />
        <div className="pt-16 flex items-center justify-center min-h-screen">
          <p className="text-gray-700 dark:text-gray-300 font-jetbrains-mono">
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

  const balance = creditsData?.balance || 0;

  const creditsErrorMessage = creditsError
    ? "We couldn’t load your credits right now. Please refresh and try again."
    : null;

  const transactionsErrorMessage = transactionsError
    ? "We couldn’t load your transactions right now. Please refresh and try again."
    : null;

  return (
    <div className="min-h-screen ui-surface">
      <Header />

      <div className="pt-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-8 py-8 sm:py-12">
          <div className="mb-8">
            <h1 className="text-3xl sm:text-4xl font-bold text-gray-900 dark:text-gray-100 mb-2 font-jetbrains-mono">
              Credits
            </h1>
            <p className="text-gray-600 dark:text-gray-300 font-jetbrains-mono">
              Purchase credits for AI staging and virtual tours.
            </p>

            {finalizeMessage ? (
              <div className="mt-4">
                <StatusBanner variant="info">{finalizeMessage}</StatusBanner>
              </div>
            ) : null}

            {creditsErrorMessage ? (
              <div className="mt-4">
                <StatusBanner variant="error" title="Credits">
                  {creditsErrorMessage}
                </StatusBanner>
              </div>
            ) : null}

            {transactionsErrorMessage ? (
              <div className="mt-4">
                <StatusBanner variant="error" title="Transactions">
                  {transactionsErrorMessage}
                </StatusBanner>
              </div>
            ) : null}
          </div>

          {/* Current Balance */}
          <div className="bg-gradient-to-br from-[var(--brand)] to-[var(--brandDark)] rounded-xl p-8 mb-8 shadow-lg">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-white/90 mb-2 font-jetbrains-mono">
                  Current Balance
                </p>
                <div className="flex items-center">
                  <Coins size={32} className="text-white mr-3" />
                  <span className="text-4xl sm:text-5xl font-bold text-white font-jetbrains-mono">
                    {creditsLoading ? "…" : balance.toLocaleString()}
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Credit Packages */}
          <div className="mb-10">
            <h2 className="text-2xl font-bold text-gray-900 dark:text-gray-100 mb-6 font-jetbrains-mono">
              Purchase Credits
            </h2>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-6">
              {CREDIT_PACKAGES.map((pkg) => (
                <div
                  key={pkg.packType}
                  className="bg-white dark:bg-[#262626] rounded-xl shadow-lg dark:shadow-none dark:ring-1 dark:ring-gray-700 p-6 hover:shadow-xl dark:hover:ring-gray-600 transition-all"
                >
                  <div className="text-center mb-6">
                    <h3 className="text-xl font-semibold text-gray-900 dark:text-gray-100 mb-2 font-jetbrains-mono">
                      {pkg.label}
                    </h3>

                    {/* credits value + "credits" on the same line */}
                    <div className="flex items-end justify-center gap-2 mb-4">
                      <Coins className="text-[var(--brand)]" size={24} />
                      <span className="text-3xl font-bold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
                        {pkg.credits}
                      </span>
                      <span className="text-[26px] font-semibold text-gray-700 dark:text-gray-300 font-jetbrains-mono leading-none">
                        credits
                      </span>
                    </div>
                  </div>

                  {/* Price shown directly above the purchase button */}
                  <div className="mb-3 text-center text-lg font-semibold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
                    {pkg.priceEur}
                  </div>

                  <button
                    onClick={() => handlePurchase(pkg.packType)}
                    className="w-full flex items-center justify-center px-6 py-3 bg-[var(--brand90)] hover:bg-[var(--brand)] text-white rounded-lg font-medium transition-colors font-jetbrains-mono"
                  >
                    <ShoppingCart size={18} className="mr-2" />
                    Purchase
                  </button>
                </div>
              ))}
            </div>
          </div>

          {/* Simple usage summary */}
          <div className="bg-white dark:bg-[#262626] rounded-xl shadow-lg dark:shadow-none dark:ring-1 dark:ring-gray-700 p-6 sm:p-8 mb-8">
            <h2 className="text-2xl font-bold text-gray-900 dark:text-gray-100 mb-4 font-jetbrains-mono">
              Usage Summary
            </h2>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div className="rounded-lg border border-gray-200 dark:border-gray-700 p-4">
                <div className="text-sm text-gray-600 dark:text-gray-300 font-jetbrains-mono">
                  Credits spent on Staging
                </div>
                <div className="mt-2 text-2xl font-bold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
                  {usageTotals.stagingSpent.toLocaleString()}
                </div>
              </div>
              <div className="rounded-lg border border-gray-200 dark:border-gray-700 p-4">
                <div className="text-sm text-gray-600 dark:text-gray-300 font-jetbrains-mono">
                  Credits spent on Virtual Tours
                </div>
                <div className="mt-2 text-2xl font-bold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
                  {usageTotals.tourSpent.toLocaleString()}
                </div>
              </div>
            </div>
            <div className="mt-3 text-xs text-gray-500 dark:text-gray-500 font-jetbrains-mono">
              Computed from your transaction history.
            </div>
          </div>

          {/* Transaction History (read-only) */}
          <div>
            <h2 className="text-2xl font-bold text-gray-900 dark:text-gray-100 mb-6 font-jetbrains-mono">
              Transaction History
            </h2>
            <div className="bg-white dark:bg-[#262626] rounded-xl shadow-lg dark:shadow-none dark:ring-1 dark:ring-gray-700 overflow-hidden">
              {transactions && transactions.length > 0 ? (
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead className="bg-gray-50 dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700">
                      <tr>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider font-jetbrains-mono">
                          Date
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider font-jetbrains-mono">
                          Type
                        </th>
                        <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider font-jetbrains-mono">
                          Credits
                        </th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                      {transactions.map((txn) => (
                        <tr key={txn.id}>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-gray-100 font-jetbrains-mono">
                            {new Date(txn.created_at).toLocaleDateString()}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm font-jetbrains-mono">
                            <span
                              className={`px-2 py-1 rounded-full text-xs ${
                                txn.transaction_type === "purchase"
                                  ? "bg-green-100 dark:bg-green-900 text-green-800 dark:text-green-200"
                                  : "bg-[var(--brandSoft)] dark:bg-[var(--brandSoftDark)] text-[var(--brandDark)] dark:text-[var(--brand)]"
                              }`}
                            >
                              {txn.transaction_type}
                            </span>
                          </td>
                          <td
                            className={`px-6 py-4 whitespace-nowrap text-sm text-right font-semibold font-jetbrains-mono ${
                              txn.credits_delta > 0
                                ? "text-green-600 dark:text-green-400"
                                : "text-red-600 dark:text-red-400"
                            }`}
                          >
                            {txn.credits_delta > 0 ? "+" : ""}
                            {txn.credits_delta}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="text-center py-12">
                  <Coins className="mx-auto mb-4 text-gray-400" size={48} />
                  <p className="text-gray-600 dark:text-gray-400 font-jetbrains-mono">
                    No transactions yet
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
