import { useState } from "react";

export default function SignInPage() {
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const onSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    if (!email || !password) {
      setError("Please fill in all fields");
      setLoading(false);
      return;
    }

    try {
      if (typeof window === "undefined") {
        throw new Error("Sign-in is only available in the browser");
      }

      // Dynamically import auth client to avoid crashing the page on load
      const authReact = await import("@auth/create/react");
      const signIn = authReact?.signIn;
      if (typeof signIn !== "function") {
        throw new Error("Auth client failed to load");
      }

      const callbackFromQuery = new URLSearchParams(window.location.search).get(
        "callbackUrl",
      );

      await signIn("credentials-signin", {
        email,
        password,
        callbackUrl: callbackFromQuery || "/properties",
        redirect: true,
      });
    } catch (err) {
      console.error("Sign-in error", err);
      setError(
        "Sign in page failed to load correctly. Please refresh, or try again in a moment.",
      );
      setLoading(false);
    }
  };

  return (
    <div className="flex min-h-screen w-full items-center justify-center bg-gray-50 dark:bg-[#1E1E1E] p-4">
      <form
        noValidate
        onSubmit={onSubmit}
        className="w-full max-w-md rounded-2xl bg-white dark:bg-[#262626] p-8 shadow-xl dark:shadow-none dark:ring-1 dark:ring-gray-700"
      >
        <h1 className="mb-8 text-center text-3xl font-bold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
          Welcome Back
        </h1>

        <div className="space-y-6">
          <div className="space-y-2">
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 font-jetbrains-mono">
              Email
            </label>
            <div className="overflow-hidden rounded-lg border border-gray-200 dark:border-gray-600 bg-white dark:bg-gray-800 px-4 py-3 focus-within:border-[var(--brand)] focus-within:ring-1 focus-within:ring-[var(--brand)]">
              <input
                required
                name="email"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
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
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full rounded-lg bg-transparent text-lg outline-none text-gray-900 dark:text-gray-100 placeholder-gray-500 dark:placeholder-gray-400 font-jetbrains-mono"
                placeholder="Enter your password"
              />
            </div>
          </div>

          {error && (
            <div className="rounded-lg bg-red-50 dark:bg-red-900/30 p-3 text-sm text-red-600 dark:text-red-400 font-jetbrains-mono">
              {error}
            </div>
          )}

          <button
            type="submit"
            disabled={loading}
            className="w-full rounded-lg bg-gray-900 dark:bg-gray-100 px-4 py-3 text-base font-medium text-white dark:text-gray-900 transition-colors hover:bg-gray-800 dark:hover:bg-gray-200 focus:outline-none focus:ring-2 focus:ring-[var(--brand)] focus:ring-offset-2 disabled:opacity-50 font-jetbrains-mono"
          >
            {loading ? "Signing in..." : "Sign In"}
          </button>
          <p className="text-center text-sm text-gray-600 dark:text-gray-400 font-jetbrains-mono">
            Don't have an account?{" "}
            <a
              href={`/account/signup${
                typeof window !== "undefined" ? window.location.search : ""
              }`}
              className="text-[var(--brandDark)] dark:text-[var(--brand)] hover:underline"
            >
              Sign up
            </a>
          </p>
        </div>
      </form>
    </div>
  );
}
