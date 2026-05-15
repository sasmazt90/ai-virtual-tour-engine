import { Coins, Menu, Moon, Sun, X } from "lucide-react";
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import useUser from "@/utils/useUser";
import { useTheme } from "@/components/ThemeProvider";

export function Header() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const { data: user } = useUser();
  const { theme, toggleTheme } = useTheme();

  const { data: creditsData } = useQuery({
    queryKey: ["credits", user?.id],
    queryFn: async () => {
      if (!user?.id) return null;
      const res = await fetch("/api/credits");
      if (!res.ok) throw new Error("Failed to fetch credits");
      return res.json();
    },
    enabled: !!user?.id,
  });

  const credits = creditsData?.balance || 0;

  const logoUrl =
    "https://ucarecdn.com/f3c2cf7c-ce51-4a0c-af7b-24d29246221e/-/format/auto/";

  return (
    <header className="fixed top-0 left-0 right-0 z-50 border-b border-white/10 bg-[#07080A]">
      <div className="max-w-7xl mx-auto px-4 sm:px-8 py-3 flex justify-between items-center h-16">
        {/* Logo */}
        <a href="/properties" className="flex items-center gap-3">
          <img
            src={logoUrl}
            alt="360 Estate Suite logo"
            className="w-9 h-9 rounded-lg border border-white/10 bg-black/20 object-contain"
            draggable={false}
          />
          <div className="text-base sm:text-lg font-semibold text-gray-50 font-jetbrains-mono tracking-tight">
            360 Estate Suite
          </div>
        </a>

        {/* Desktop Navigation */}
        <div className="hidden md:flex space-x-6 lg:space-x-8 items-center">
          <a
            href="/properties"
            className="text-gray-300 hover:text-white transition-colors font-jetbrains-mono"
          >
            Properties
          </a>
          <a
            href="/directory"
            className="text-gray-300 hover:text-white transition-colors font-jetbrains-mono"
          >
            Directory
          </a>
          <a
            href="/calendar"
            className="text-gray-300 hover:text-white transition-colors font-jetbrains-mono"
          >
            Calendar
          </a>
          <a
            href="/credits"
            className="flex items-center space-x-2 text-gray-300 hover:text-white transition-colors font-jetbrains-mono"
          >
            <Coins size={18} className="text-[var(--brand)]" />
            <span>{credits.toLocaleString()}</span>
          </a>
          <a
            href="/profile"
            className="text-gray-300 hover:text-white transition-colors font-jetbrains-mono"
          >
            Profile
          </a>

          <button
            type="button"
            onClick={toggleTheme}
            className="inline-flex items-center justify-center rounded-lg border border-white/10 bg-white/5 hover:bg-white/10 px-2.5 py-2 transition-colors"
            aria-label={
              theme === "dark"
                ? "Switch to light theme"
                : "Switch to dark theme"
            }
            title={theme === "dark" ? "Light theme" : "Dark theme"}
          >
            {theme === "dark" ? (
              <Sun size={18} className="text-gray-100" />
            ) : (
              <Moon size={18} className="text-gray-100" />
            )}
          </button>
        </div>

        {/* Mobile menu button */}
        <div className="flex items-center gap-2 md:hidden">
          <button
            type="button"
            onClick={toggleTheme}
            className="p-2 rounded-lg border border-white/10 bg-white/5 hover:bg-white/10 transition-colors"
            aria-label={
              theme === "dark"
                ? "Switch to light theme"
                : "Switch to dark theme"
            }
          >
            {theme === "dark" ? (
              <Sun size={18} className="text-gray-100" />
            ) : (
              <Moon size={18} className="text-gray-100" />
            )}
          </button>

          <button
            className="p-1 text-gray-200 hover:text-white transition-colors"
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            aria-label={mobileMenuOpen ? "Close menu" : "Open menu"}
          >
            {mobileMenuOpen ? <X size={24} /> : <Menu size={24} />}
          </button>
        </div>
      </div>

      {/* Mobile Navigation */}
      {mobileMenuOpen && (
        <div className="md:hidden border-t border-white/10 bg-[#07080A]">
          <div className="px-4 py-4 space-y-3">
            <a
              href="/properties"
              className="block text-gray-300 hover:text-white transition-colors font-jetbrains-mono"
            >
              Properties
            </a>
            <a
              href="/directory"
              className="block text-gray-300 hover:text-white transition-colors font-jetbrains-mono"
            >
              Directory
            </a>
            <a
              href="/calendar"
              className="block text-gray-300 hover:text-white transition-colors font-jetbrains-mono"
            >
              Calendar
            </a>
            <a
              href="/credits"
              className="flex items-center space-x-2 text-gray-300 hover:text-white transition-colors font-jetbrains-mono"
            >
              <Coins size={18} className="text-[var(--brand)]" />
              <span>Credits: {credits.toLocaleString()}</span>
            </a>
            <a
              href="/profile"
              className="block text-gray-300 hover:text-white transition-colors font-jetbrains-mono"
            >
              Profile
            </a>
          </div>
        </div>
      )}
    </header>
  );
}
