import type { Metadata } from "next";
import type { ReactNode } from "react";
import { Montserrat } from "next/font/google";

import "./globals.css";
import { AppShell } from "@/components/app-shell";
import { cn } from "@/lib/utils";

const montserrat = Montserrat({
  subsets: ["latin"],
  variable: "--font-montserrat",
  weight: ["400", "500", "600", "700"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "Synaptic Tuner",
  description: "Manage datasets, training, evaluation, and local models for the Obsidian Nexus toolset.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={cn("min-h-screen bg-background font-sans antialiased", montserrat.variable)}>
        <AppShell>{children}</AppShell>
      </body>
    </html>
  );
}
