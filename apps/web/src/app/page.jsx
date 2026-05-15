import { useEffect, useMemo } from "react";
import MarketingHeader from "@/components/MarketingHeader";
import useMarketingAnalytics from "@/hooks/useMarketingAnalytics";
import { HeroSection } from "@/components/HomePage/HeroSection";
import { FeaturesSection } from "@/components/HomePage/FeaturesSection";
import { ShowcaseSection } from "@/components/HomePage/ShowcaseSection";
import { FinalCTASection } from "@/components/HomePage/FinalCTASection";
import { HomePageStyles } from "@/components/HomePage/HomePageStyles";

export default function HomePage() {
  const { trackSignUpClick, trackPricingClick } = useMarketingAnalytics();

  useEffect(() => {
    if (typeof window !== "undefined") {
      document.title = "360 Estate Suite";
    }
  }, []);

  const onHeroSignUpClick = useMemo(() => {
    return () => trackSignUpClick("home_hero");
  }, [trackSignUpClick]);

  const onBottomSignUpClick = useMemo(() => {
    return () => trackSignUpClick("home_bottom_cta");
  }, [trackSignUpClick]);

  const onPricingClick = useMemo(() => {
    return () => trackPricingClick("home_final_cta");
  }, [trackPricingClick]);

  return (
    <div className="min-h-screen text-gray-900 dark:text-gray-100">
      <MarketingHeader />

      <main>
        <HeroSection onSignUpClick={onHeroSignUpClick} />
        <FeaturesSection />
        <ShowcaseSection />
        <FinalCTASection
          onSignUpClick={onBottomSignUpClick}
          onPricingClick={onPricingClick}
        />
      </main>

      <HomePageStyles />
    </div>
  );
}
