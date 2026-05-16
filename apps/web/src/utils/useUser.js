import * as React from "react";

const useUser = () => {
  const [user, setUser] = React.useState(null);
  const [loading, setLoading] = React.useState(true);

  const fetchUser = React.useCallback(async () => {
    const res = await fetch("/api/auth/session", {
      credentials: "include",
      cache: "no-store",
    });
    if (!res.ok) return null;
    const session = await res.json().catch(() => null);
    return session?.user ?? null;
  }, []);

  const refetchUser = React.useCallback(async () => {
    setLoading(true);
    try {
      let nextUser = null;
      for (let attempt = 0; attempt < 4; attempt += 1) {
        nextUser = await fetchUser();
        if (nextUser) break;
        if (attempt < 3) {
          await new Promise((resolve) => setTimeout(resolve, 350));
        }
      }
      setUser(nextUser);
    } catch {
      setUser(null);
    } finally {
      setLoading(false);
    }
  }, [fetchUser]);

  React.useEffect(refetchUser, [refetchUser]);

  // ---
  // SSR hydration safety:
  // During server render we don't want pages to branch into "unauthenticated" UI
  // (or `return null`) and then render the real app immediately on the client.
  // That pattern triggers hard hydration mismatches.
  // So on the server we always report `loading: true`.
  // ---
  const isServer = typeof window === "undefined";

  if (process.env.NEXT_PUBLIC_CREATE_ENV !== "PRODUCTION") {
    return {
      user,
      data: isServer ? null : user,
      loading: isServer ? true : loading,
      refetch: refetchUser,
    };
  }
  return {
    user,
    data: isServer ? null : user,
    loading: isServer ? true : loading,
    refetch: refetchUser,
  };
};

export { useUser };

export default useUser;
