import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Sports Pick — NBA Edge Finder",
  description: "Elo-driven NBA moneyline picks, parlays, and AI bet analysis.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-surface text-white antialiased">{children}</body>
    </html>
  );
}
