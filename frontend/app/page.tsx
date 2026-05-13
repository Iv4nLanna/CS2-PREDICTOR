import { api } from "@/lib/api";
import { MatchCard } from "@/components/MatchCard";

export const revalidate = 60;

export default async function HomePage() {
  let matches: Awaited<ReturnType<typeof api.upcomingMatches>> = [];
  let errored = false;
  try {
    matches = await api.upcomingMatches();
  } catch {
    errored = true;
  }

  return (
    <div className="mx-auto max-w-3xl">
      <h1 className="mb-6 text-2xl font-semibold">Próximas partidas</h1>
      {errored && (
        <p className="rounded border border-rose-800 bg-rose-950/40 px-4 py-3 text-sm text-rose-200">
          Não foi possível carregar as partidas no momento.
        </p>
      )}
      {!errored && matches.length === 0 && (
        <p className="text-zinc-400">Nenhuma partida agendada com previsão disponível.</p>
      )}
      <div className="space-y-3">
        {matches.map((m) => <MatchCard key={m.match_id} match={m} />)}
      </div>
    </div>
  );
}
