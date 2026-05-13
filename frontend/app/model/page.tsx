import { api } from "@/lib/api";

export const revalidate = 600;

export default async function ModelPage() {
  let runs: Awaited<ReturnType<typeof api.modelAccuracy>> = [];
  try {
    runs = await api.modelAccuracy();
  } catch {
    return <p className="text-rose-300">Não foi possível carregar o histórico do modelo.</p>;
  }

  const current = runs[0];

  return (
    <div className="mx-auto max-w-3xl">
      <h1 className="mb-2 text-2xl font-semibold">Transparência do modelo</h1>
      <p className="mb-6 text-sm text-zinc-400">
        Cada execução do pipeline retreina o modelo e registra a acurácia em validação temporal.
        As probabilidades são calibradas via Platt Scaling para refletir frequências reais.
      </p>

      {current && (
        <div className="mb-8 rounded-lg border border-zinc-800 bg-zinc-900 p-4">
          <p className="text-xs text-zinc-400">Versão atual</p>
          <p className="text-xl font-semibold">{current.version}</p>
          <p className="mt-2 text-sm">
            Acurácia: <span className="font-semibold">{(current.accuracy * 100).toFixed(1)}%</span>
          </p>
          <p className="text-xs text-zinc-500">
            Treinado em {new Date(current.trained_at).toLocaleString("pt-BR")}
          </p>
          <p className="mt-2 text-xs text-zinc-400">
            Features: {current.features_used.join(", ")}
          </p>
        </div>
      )}

      <h2 className="mb-3 text-lg font-semibold">Histórico de versões</h2>
      <table className="w-full text-sm">
        <thead>
          <tr className="text-left text-zinc-400">
            <th className="py-2">Versão</th>
            <th className="py-2">Treinado em</th>
            <th className="py-2">Acurácia</th>
            <th className="py-2"># Features</th>
          </tr>
        </thead>
        <tbody>
          {runs.map((r) => (
            <tr key={r.version} className="border-t border-zinc-800">
              <td className="py-2">{r.version}</td>
              <td className="py-2 text-zinc-400">
                {new Date(r.trained_at).toLocaleString("pt-BR")}
              </td>
              <td className="py-2">{(r.accuracy * 100).toFixed(1)}%</td>
              <td className="py-2 text-zinc-400">{r.features_used.length}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
