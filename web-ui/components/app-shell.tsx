'use client';

import type { ComponentType, ReactNode } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { ThemeProvider } from "next-themes";
import { Toaster } from "sonner";
import {
  BrainCircuit,
  Database,
  HardDrive,
  LayoutDashboard,
  LineChart,
  Settings,
} from "lucide-react";

import { cn } from "@/lib/utils";

type NavItem = {
  label: string;
  href: string;
  icon: ComponentType<{ className?: string }>;
};

const navItems: NavItem[] = [
  { label: "Dashboard", href: "/", icon: LayoutDashboard },
  { label: "Datasets", href: "/datasets", icon: Database },
  { label: "Training", href: "/training", icon: BrainCircuit },
  { label: "Evaluation", href: "/evaluation", icon: LineChart },
  { label: "Models", href: "/models", icon: HardDrive },
  { label: "Settings", href: "/settings", icon: Settings },
];

interface AppShellProps {
  children: ReactNode;
}

export function AppShell({ children }: AppShellProps) {
  const pathname = usePathname();

  return (
    <ThemeProvider attribute="class" defaultTheme="light" enableSystem>
      <div className="min-h-screen bg-gradient-to-br from-primary/5 via-background to-secondary/5 text-foreground">
        <div className="mx-auto max-w-7xl px-4 py-6 lg:px-8">
          <div className="grid gap-6 lg:grid-cols-[260px,1fr]">
            <aside className="rounded-2xl border bg-card/80 shadow-sm backdrop-blur">
              <div className="space-y-6 p-6">
                <Link href="/" className="flex items-center gap-3">
                  <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-primary/10 text-sm font-semibold uppercase text-primary">
                    ST
                  </div>
                  <div className="leading-tight">
                    <p className="text-xs font-medium tracking-[0.18em] text-muted-foreground">
                      Synaptic
                    </p>
                    <p className="text-lg font-semibold">Tuner</p>
                  </div>
                </Link>

                <nav className="flex gap-2 overflow-x-auto pb-1 lg:flex-col lg:gap-1 lg:pb-0">
                  {navItems.map((item) => {
                    const isActive =
                      pathname === item.href ||
                      (item.href !== "/" && pathname?.startsWith(item.href));

                    return (
                      <Link
                        key={item.href}
                        href={item.href}
                        aria-current={isActive ? "page" : undefined}
                        className={cn(
                          "group flex min-w-[140px] items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors lg:min-w-0",
                          isActive
                            ? "bg-primary/10 text-primary ring-1 ring-primary/20"
                            : "text-muted-foreground hover:bg-muted/60 hover:text-foreground"
                        )}
                      >
                        <item.icon className="h-4 w-4" />
                        <span>{item.label}</span>
                      </Link>
                    );
                  })}
                </nav>

                <div className="rounded-lg border border-dashed border-muted-foreground/30 bg-muted/50 p-3 text-xs text-muted-foreground">
                  Manage datasets, training, evaluation, and deployments for your
                  local stack.
                </div>
              </div>
            </aside>

            <main className="rounded-2xl border bg-card/90 shadow-sm backdrop-blur">
              <div className="p-6 lg:p-8">{children}</div>
            </main>
          </div>
        </div>
      </div>

      <Toaster position="top-right" richColors closeButton />
    </ThemeProvider>
  );
}
