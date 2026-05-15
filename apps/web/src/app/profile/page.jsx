import { Header } from "../../components/Header";
import { useEffect, useMemo, useState, useCallback } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import useUser from "@/utils/useUser";
import useUpload from "@/utils/useUpload";

export default function ProfilePage() {
  const { data: user, loading: userLoading } = useUser();
  const [fullName, setFullName] = useState("");
  const [company, setCompany] = useState("");
  const [companyLogoUrl, setCompanyLogoUrl] = useState("");
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState(null);

  const [upload, { loading: logoUploading }] = useUpload();

  const isAdmin = useMemo(() => {
    const email = String(user?.email || "").toLowerCase();
    return email === "sasmazt90@gmail.com";
  }, [user?.email]);

  const profileQuery = useQuery({
    queryKey: ["profile", user?.id],
    queryFn: async () => {
      const res = await fetch("/api/profile");
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body?.error || "Failed to fetch profile");
      }
      return res.json();
    },
    enabled: !!user?.id,
  });

  useEffect(() => {
    if (!profileQuery.data) return;

    // Populate the form once (do not wipe user edits while typing).
    setFullName((prev) =>
      prev ? prev : String(profileQuery.data?.full_name || ""),
    );
    setCompany((prev) =>
      prev ? prev : String(profileQuery.data?.company || ""),
    );
    setCompanyLogoUrl((prev) =>
      prev ? prev : String(profileQuery.data?.company_logo_url || ""),
    );
  }, [profileQuery.data]);

  const saveProfileMutation = useMutation({
    mutationFn: async ({ nextFullName, nextCompany, nextLogoUrl }) => {
      const res = await fetch("/api/profile", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          full_name: nextFullName,
          company: nextCompany,
          company_logo_url: nextLogoUrl || null,
        }),
      });

      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body?.error || "Failed to update profile");
      }

      return res.json();
    },
    onSuccess: () => {
      setSaved(true);
      setError(null);
      setTimeout(() => setSaved(false), 2500);
      profileQuery.refetch();
    },
    onError: (e) => {
      console.error(e);
      setError(e?.message || "Failed to update profile");
    },
  });

  const handleSave = useCallback(() => {
    setSaved(false);
    setError(null);
    saveProfileMutation.mutate({
      nextFullName: fullName,
      nextCompany: company,
      nextLogoUrl: companyLogoUrl,
    });
  }, [company, companyLogoUrl, fullName, saveProfileMutation]);

  const onPickLogo = useCallback(
    async (e) => {
      try {
        setError(null);
        const file = e.target.files?.[0] || null;
        if (!file) return;

        const { url, error: uploadError, mimeType } = await upload({ file });
        if (uploadError) {
          throw new Error(uploadError);
        }
        if (!mimeType || !String(mimeType).startsWith("image/")) {
          throw new Error("Please upload an image file");
        }

        setCompanyLogoUrl(url);
      } catch (err) {
        console.error(err);
        setError(err?.message || "Could not upload logo");
      }
    },
    [upload],
  );

  const {
    data: adminUsers,
    isLoading: adminUsersLoading,
    error: adminUsersError,
  } = useQuery({
    queryKey: ["adminUsers"],
    queryFn: async () => {
      const res = await fetch("/api/admin/users");
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body?.error || "Failed to fetch users");
      }
      return res.json();
    },
    enabled: !!user?.id && isAdmin,
  });

  const [targetUserId, setTargetUserId] = useState("");
  const [creditsToGrant, setCreditsToGrant] = useState("");
  const [grantMessage, setGrantMessage] = useState(null);
  const [grantError, setGrantError] = useState(null);

  const grantMutation = useMutation({
    mutationFn: async () => {
      setGrantError(null);
      setGrantMessage(null);

      const credits = Math.trunc(Number(creditsToGrant || 0));
      if (!Number.isFinite(credits) || credits <= 0) {
        throw new Error("Please enter a positive credits amount");
      }

      if (!targetUserId) {
        throw new Error("Please select a user");
      }

      const res = await fetch("/api/credits/grant", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          targetUserId,
          credits,
          reason: "admin_grant",
        }),
      });

      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body?.error || "Could not grant credits");
      }

      return res.json();
    },
    onSuccess: (data) => {
      const balance = Number(data?.balance || 0);
      setGrantMessage(
        `Credits granted. New balance: ${balance.toLocaleString()}`,
      );
      setCreditsToGrant("");
    },
    onError: (e) => {
      console.error(e);
      setGrantError(e?.message || "Could not grant credits");
    },
  });

  if (userLoading) {
    return (
      <div className="min-h-screen">
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

  const adminSelectOptions = Array.isArray(adminUsers) ? adminUsers : [];

  return (
    <div className="min-h-screen">
      <Header />

      <div className="pt-16">
        <div className="max-w-4xl mx-auto px-4 sm:px-8 py-8 sm:py-12">
          <div className="mb-8">
            <h1 className="text-3xl sm:text-4xl font-bold text-gray-900 dark:text-gray-100 mb-2 font-jetbrains-mono">
              Profile
            </h1>
            <p className="text-gray-600 dark:text-gray-300 font-jetbrains-mono">
              Manage your account settings
            </p>
          </div>

          <div className="bg-white/70 dark:bg-white/5 rounded-xl border border-black/10 dark:border-white/10 backdrop-blur shadow-[0_14px_60px_rgba(0,0,0,0.18)] p-6 sm:p-8 mb-6">
            <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100 mb-6 font-jetbrains-mono">
              Account Information
            </h2>

            {profileQuery.isLoading ? (
              <div className="mb-4 text-sm text-gray-600 dark:text-gray-300 font-jetbrains-mono">
                Loading profile…
              </div>
            ) : null}

            {error ? (
              <div className="mb-4 text-sm text-red-600 dark:text-red-400 font-jetbrains-mono">
                {error}
              </div>
            ) : null}

            <div className="space-y-6">
              <div className="space-y-2">
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 font-jetbrains-mono">
                  Email
                </label>
                <div className="px-4 py-3 bg-gray-100 dark:bg-gray-700 border border-gray-200 dark:border-gray-600 rounded-lg text-gray-600 dark:text-gray-400 font-jetbrains-mono">
                  {user.email}
                </div>
                <p className="text-xs text-gray-500 dark:text-gray-500 font-jetbrains-mono">
                  Email cannot be changed
                </p>
              </div>

              <div className="space-y-2">
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 font-jetbrains-mono">
                  Full Name
                </label>
                <input
                  type="text"
                  value={fullName}
                  onChange={(e) => setFullName(e.target.value)}
                  placeholder={user.name || "Enter your full name"}
                  className="w-full px-4 py-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg text-gray-900 dark:text-gray-100 placeholder-gray-500 dark:placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-[var(--brand)] font-jetbrains-mono"
                />
              </div>

              <div className="space-y-2">
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 font-jetbrains-mono">
                  Company
                </label>
                <input
                  type="text"
                  value={company}
                  onChange={(e) => setCompany(e.target.value)}
                  placeholder="Enter your company name"
                  className="w-full px-4 py-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg text-gray-900 dark:text-gray-100 placeholder-gray-500 dark:placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-[var(--brand)] font-jetbrains-mono"
                />
              </div>

              {/* NEW: Company logo upload */}
              <div className="space-y-2">
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 font-jetbrains-mono">
                  Company Logo
                </label>

                {companyLogoUrl ? (
                  <div className="flex items-center gap-3">
                    <img
                      src={companyLogoUrl}
                      alt="Company logo"
                      className="h-12 w-12 rounded-md object-cover border border-gray-200 dark:border-gray-600"
                    />
                    <div className="text-xs text-gray-600 dark:text-gray-300 font-jetbrains-mono break-all">
                      {companyLogoUrl}
                    </div>
                  </div>
                ) : (
                  <div className="text-xs text-gray-500 dark:text-gray-400 font-jetbrains-mono">
                    Upload a logo to show it on generated contracts.
                  </div>
                )}

                <input
                  type="file"
                  accept="image/*"
                  onChange={onPickLogo}
                  disabled={logoUploading}
                  className="block w-full text-sm text-gray-700 dark:text-gray-300 font-jetbrains-mono file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-jetbrains-mono file:bg-[var(--brandSoft)] file:text-[var(--brandDark)] hover:file:bg-[var(--brandSoftDark)]"
                />

                {logoUploading ? (
                  <div className="text-xs text-gray-500 dark:text-gray-400 font-jetbrains-mono">
                    Uploading…
                  </div>
                ) : null}
              </div>

              <div className="flex gap-4 pt-4">
                <button
                  onClick={handleSave}
                  disabled={saveProfileMutation.isPending}
                  className="px-6 py-3 bg-[var(--brand90)] hover:bg-[var(--brand)] text-white rounded-lg font-medium transition-colors font-jetbrains-mono disabled:opacity-50"
                >
                  {saveProfileMutation.isPending ? "Saving…" : "Save Changes"}
                </button>
                {saved && (
                  <div className="flex items-center text-green-600 dark:text-green-400 font-jetbrains-mono">
                    Profile updated successfully!
                  </div>
                )}
              </div>
            </div>
          </div>

          <div className="bg-white/70 dark:bg-white/5 rounded-xl border border-black/10 dark:border-white/10 backdrop-blur shadow-[0_14px_60px_rgba(0,0,0,0.18)] p-6 sm:p-8">
            <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100 mb-4 font-jetbrains-mono">
              Sign Out
            </h2>
            <p className="text-gray-600 dark:text-gray-300 mb-6 font-jetbrains-mono">
              Sign out of your account on this device
            </p>
            <a
              href="/account/logout"
              className="inline-block px-6 py-3 bg-red-600 dark:bg-red-700 text-white rounded-lg font-medium hover:bg-red-700 dark:hover:bg-red-800 transition-colors font-jetbrains-mono"
            >
              Sign Out
            </a>
          </div>

          {/* Admin-only: Add Credits */}
          {isAdmin ? (
            <div className="mt-6 bg-white/70 dark:bg-white/5 rounded-xl border border-black/10 dark:border-white/10 backdrop-blur shadow-[0_14px_60px_rgba(0,0,0,0.18)] p-6 sm:p-8">
              <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100 mb-2 font-jetbrains-mono">
                Admin: Add Credits
              </h2>
              <p className="text-sm text-gray-600 dark:text-gray-300 mb-5 font-jetbrains-mono">
                Select a user and grant credits to their account.
              </p>

              {adminUsersError ? (
                <div className="mb-4 text-sm text-red-600 dark:text-red-400 font-jetbrains-mono">
                  {adminUsersError?.message || "Failed to load users"}
                </div>
              ) : null}

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 font-jetbrains-mono">
                    User
                  </label>
                  <select
                    value={targetUserId}
                    onChange={(e) => setTargetUserId(e.target.value)}
                    className="w-full px-4 py-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-[var(--brand)] font-jetbrains-mono"
                  >
                    <option value="">Select a user…</option>
                    {adminUsersLoading
                      ? null
                      : adminSelectOptions.map((u) => {
                          const label = u?.name
                            ? `${u.name} • ${u.email}`
                            : String(u?.email || u?.user_id);
                          return (
                            <option key={u.user_id} value={u.user_id}>
                              {label}
                            </option>
                          );
                        })}
                  </select>
                </div>

                <div className="space-y-2">
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 font-jetbrains-mono">
                    Credits
                  </label>
                  <input
                    value={creditsToGrant}
                    onChange={(e) => setCreditsToGrant(e.target.value)}
                    inputMode="numeric"
                    placeholder="e.g. 100"
                    className="w-full px-4 py-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg text-gray-900 dark:text-gray-100 placeholder-gray-500 dark:placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-[var(--brand)] font-jetbrains-mono"
                  />
                </div>
              </div>

              {grantError ? (
                <div className="mt-4 text-sm text-red-600 dark:text-red-400 font-jetbrains-mono">
                  {grantError}
                </div>
              ) : null}

              {grantMessage ? (
                <div className="mt-4 text-sm text-green-700 dark:text-green-300 font-jetbrains-mono">
                  {grantMessage}
                </div>
              ) : null}

              <div className="mt-5 flex justify-end">
                <button
                  type="button"
                  onClick={() => grantMutation.mutate()}
                  disabled={grantMutation.isPending}
                  className="px-6 py-3 bg-[var(--brand90)] hover:bg-[var(--brand)] text-white rounded-lg font-medium transition-colors font-jetbrains-mono disabled:opacity-50"
                >
                  {grantMutation.isPending ? "Granting…" : "Add Credits"}
                </button>
              </div>
            </div>
          ) : null}

          {/* Internal tools (admin-only) */}
          {isAdmin ? (
            <div className="mt-6 bg-white/70 dark:bg-white/5 rounded-xl border border-black/10 dark:border-white/10 backdrop-blur shadow-[0_14px_60px_rgba(0,0,0,0.18)] p-6 sm:p-8">
              <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100 mb-4 font-jetbrains-mono">
                Internal Tools
              </h2>
              <p className="text-gray-600 dark:text-gray-300 mb-4 font-jetbrains-mono">
                Read-only support/debug views.
              </p>
              <div className="flex flex-col sm:flex-row gap-3">
                <a
                  href="/profile/tools"
                  className="inline-flex items-center justify-center px-6 py-3 bg-[var(--brand90)] hover:bg-[var(--brand)] text-white rounded-lg font-medium transition-colors font-jetbrains-mono"
                >
                  Config
                </a>
                <a
                  href="/profile/ai-jobs"
                  className="inline-flex items-center justify-center px-6 py-3 bg-white dark:bg-[#262626] border border-gray-200 dark:border-gray-700 text-gray-900 dark:text-gray-100 rounded-lg font-medium hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors font-jetbrains-mono"
                >
                  AI Jobs
                </a>
              </div>
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}
