type Props = {
  leftLabel: string;
  rightLabel: string;
  leftProb: number;
};

export function ProbabilityBar({ leftLabel, rightLabel, leftProb }: Props) {
  const leftPct = Math.round(leftProb * 100);
  const rightPct = 100 - leftPct;
  return (
    <div className="w-full">
      <div className="mb-1 flex justify-between text-xs text-zinc-400">
        <span>{leftLabel}</span>
        <span>{rightLabel}</span>
      </div>
      <div className="flex h-3 w-full overflow-hidden rounded bg-zinc-800">
        <div className="bg-emerald-500" style={{ width: `${leftPct}%` }} />
        <div className="bg-rose-500" style={{ width: `${rightPct}%` }} />
      </div>
      <div className="mt-1 flex justify-between text-sm font-medium">
        <span>{leftPct}%</span>
        <span>{rightPct}%</span>
      </div>
    </div>
  );
}
