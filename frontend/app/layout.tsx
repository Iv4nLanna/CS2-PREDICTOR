import "./globals.css";
import type { Metadata, Viewport } from "next";

export const metadata: Metadata = {
  title: "CS2 Win Predictor",
  description: "Probabilidade estatística de vitória em partidas profissionais de CS2.",
};

export const viewport: Viewport = { themeColor: "#0a0a0a" };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="pt-BR">
      <body className="min-h-screen">
        <header className="border-b border-zinc-800 px-6 py-4">
          <a href="/" className="text-lg font-semibold">CS2 Win Predictor</a>
          <nav className="ml-6 inline-flex gap-4 text-sm text-zinc-400">
            <a href="/teams" className="hover:text-zinc-100">Times</a>
            <a href="/model" className="hover:text-zinc-100">Modelo</a>
          </nav>
        </header>
        <main className="px-6 py-8">{children}</main>
      </body>
    </html>
  );
}
