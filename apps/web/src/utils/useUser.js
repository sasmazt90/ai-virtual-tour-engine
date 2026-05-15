import * as React from "react";
import { useSession } from "@auth/create/react";

const useUser = () => {
  const { data: session, status } = useSession();
  const id = session?.user?.id;

  const [user, setUser] = React.useState(session?.user ?? null);

  const fetchUser = React.useCallback(async (session) => {
    return session?.user;
  }, []);

  const refetchUser = React.useCallback(() => {
    if (process.env.NEXT_PUBLIC_CREATE_ENV === "PRODUCTION") {
      if (id) {
        fetchUser(session).then(setUser);
      } else {
        setUser(null);
      }
    }
  }, [fetchUser, id]);

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
      data: isServer ? null : session?.user || null,
      loading: isServer ? true : status === "loading",
      refetch: refetchUser,
    };
  }
  return {
    user,
    data: isServer ? null : user,
    loading: isServer
      ? true
      : status === "loading" || (status === "authenticated" && !user),
    refetch: refetchUser,
  };
};

export { useUser };

export default useUser;
