export default function SignUpPage() {
  const search =
    typeof window !== "undefined"
      ? new URLSearchParams(window.location.search)
      : new URLSearchParams();
  const callbackUrl = search.get("callbackUrl") || "/properties";
  const error = search.get("error");

  return (
    <div className="flex min-h-screen w-full items-center justify-center bg-gray-50 dark:bg-[#1E1E1E] p-4">
      <form
        noValidate
        method="post"
        action="/api/auth/callback/credentials-signup"
        className="w-full max-w-md rounded-2xl bg-white dark:bg-[#262626] p-8 shadow-xl dark:shadow-none dark:ring-1 dark:ring-gray-700"
      >
        <input type="hidden" name="callbackUrl" value={callbackUrl} />
        <h1 className="mb-8 text-center text-3xl font-bold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
          Create Account
        </h1>

        <div className="space-y-6">
          <div className="space-y-2">
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 font-jetbrains-mono">
              Full Name
            </label>
            <div className="overflow-hidden rounded-lg border border-gray-200 dark:border-gray-600 bg-white dark:bg-gray-800 px-4 py-3 focus-within:border-[var(--brand)] focus-within:ring-1 focus-within:ring-[var(--brand)]">
              <input
                name="name"
                type="text"
                placeholder="Enter your full name"
                className="w-full bg-transparent text-lg outline-none text-gray-900 dark:text-gray-100 placeholder-gray-500 dark:placeholder-gray-400 font-jetbrains-mono"
              />
            </div>
          </div>
          <div className="space-y-2">
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 font-jetbrains-mono">
              Email
            </label>
            <div className="overflow-hidden rounded-lg border border-gray-200 dark:border-gray-600 bg-white dark:bg-gray-800 px-4 py-3 focus-within:border-[var(--brand)] focus-within:ring-1 focus-within:ring-[var(--brand)]">
              <input
                required
                name="email"
                type="email"
                placeholder="Enter your email"
                className="w-full bg-transparent text-lg outline-none text-gray-900 dark:text-gray-100 placeholder-gray-500 dark:placeholder-gray-400 font-jetbrains-mono"
              />
            </div>
          </div>
          <div className="space-y-2">
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 font-jetbrains-mono">
              Password
            </label>
            <div className="overflow-hidden rounded-lg border border-gray-200 dark:border-gray-600 bg-white dark:bg-gray-800 px-4 py-3 focus-within:border-[var(--brand)] focus-within:ring-1 focus-within:ring-[var(--brand)]">
              <input
                required
                name="password"
                type="password"
                className="w-full rounded-lg bg-transparent text-lg outline-none text-gray-900 dark:text-gray-100 placeholder-gray-500 dark:placeholder-gray-400 font-jetbrains-mono"
                placeholder="Create a password"
              />
            </div>
          </div>

          {error && (
            <div className="rounded-lg bg-red-50 dark:bg-red-900/30 p-3 text-sm text-red-600 dark:text-red-400 font-jetbrains-mono">
              Could not create an account with these details.
            </div>
          )}

          <button
            type="submit"
            className="w-full rounded-lg bg-gray-900 dark:bg-gray-100 px-4 py-3 text-base font-medium text-white dark:text-gray-900 transition-colors hover:bg-gray-800 dark:hover:bg-gray-200 focus:outline-none focus:ring-2 focus:ring-[var(--brand)] focus:ring-offset-2 disabled:opacity-50 font-jetbrains-mono"
          >
            Sign Up
          </button>
          <p className="text-center text-sm text-gray-600 dark:text-gray-400 font-jetbrains-mono">
            Already have an account?{" "}
            <a
              href={`/account/signin${
                typeof window !== "undefined" ? window.location.search : ""
              }`}
              className="text-[var(--brandDark)] dark:text-[var(--brand)] hover:underline"
            >
              Sign in
            </a>
          </p>
        </div>
      </form>
    </div>
  );
}
