import { useQuery } from "@tanstack/react-query";

export function useCredits(userId) {
  const { data: creditsData } = useQuery({
    queryKey: ["credits", userId],
    queryFn: async () => {
      const res = await fetch("/api/credits");
      if (!res.ok) {
        throw new Error(
          `When fetching /api/credits, the response was [${res.status}] ${res.statusText}`,
        );
      }
      return res.json();
    },
    enabled: !!userId,
  });

  const creditsBalance = creditsData?.balance || 0;

  return { creditsBalance, creditsData };
}
