import useAuth from "@/utils/useAuth";

export default function LogoutPage() {
  const { signOut } = useAuth();

  const handleSignOut = async () => {
    await signOut({
      callbackUrl: "/account/signin",
      redirect: true,
    });
  };

  return (
    <div className="flex min-h-screen w-full items-center justify-center bg-gray-50 dark:bg-[#1E1E1E] p-4">
      <div className="w-full max-w-md rounded-2xl bg-white dark:bg-[#262626] p-8 shadow-xl dark:shadow-none dark:ring-1 dark:ring-gray-700">
        <h1 className="mb-8 text-center text-3xl font-bold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
          Sign Out
        </h1>

        <button
          onClick={handleSignOut}
          className="w-full rounded-lg bg-gray-900 dark:bg-gray-100 px-4 py-3 text-base font-medium text-white dark:text-gray-900 transition-colors hover:bg-gray-800 dark:hover:bg-gray-200 focus:outline-none focus:ring-2 focus:ring-[var(--brand)] focus:ring-offset-2 disabled:opacity-50 font-jetbrains-mono"
        >
          Sign Out
        </button>
      </div>
    </div>
  );
}
