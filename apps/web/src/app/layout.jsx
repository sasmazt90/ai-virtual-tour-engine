import { useState } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import MarketingFooter from "@/components/MarketingFooter";
import { ThemeProvider } from "@/components/ThemeProvider";
import { HomePageStyles } from "@/components/HomePage/HomePageStyles";

export default function RootLayout({ children }) {
  const [queryClient] = useState(() => {
    return new QueryClient({
      defaultOptions: {
        queries: {
          staleTime: 1000 * 60 * 5, // 5 minutes
          // react-query v5 uses `gcTime` (older versions used `cacheTime`).
          // Keeping this config SSR-safe is more important than the exact name.
          gcTime: 1000 * 60 * 30, // 30 minutes
          retry: 1,
          refetchOnWindowFocus: false,
        },
      },
    });
  });

  const brandStyleVars = {
    // Brand color: user-specified
    "--brand": "#eca42b",
    "--brandHover": "#d6911e",
    "--brandDark": "#d6911e",

    // Slightly transparent fills for premium UI
    "--brand90": "rgba(236, 164, 43, 0.90)",
    "--brandHover90": "rgba(214, 145, 30, 0.95)",
    "--brand70": "rgba(236, 164, 43, 0.70)",

    // Soft pill / highlight backgrounds (non-button accents)
    "--brandSoft": "rgba(236, 164, 43, 0.12)",
    "--brandSoftDark": "rgba(236, 164, 43, 0.18)",
  };

  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider brandStyleVars={brandStyleVars}>
        {children}
        <MarketingFooter />
        <HomePageStyles />
      </ThemeProvider>
    </QueryClientProvider>
  );
}
