import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { QualityRow } from "../api";
import { useAudit } from "../audit-context";
import { axisLine, axisTick, ChartTip, gridProps } from "../components/charts";
import { Badge, Card, DataTable, Dot, Page, Status, type Col } from "../components/ui";
import { ISSUE_COLOR, int, pct } from "../lib";

export function Quality() {
  const a = useAudit();
  const rows = a.quality;
  const cols: Col<QualityRow>[] = [
    { key: "dim", header: "Dimension", cell: (r) => <span className="flex items-center gap-2 font-medium"><Dot color={ISSUE_COLOR[r.issue_key] ?? "var(--muted)"} />{r.dimension}</span> },
    { key: "type", header: "Detector", cell: (r) => <Badge tone={r.detector_type === "exact check" ? "good" : "neutral"}>{r.detector_type}</Badge> },
    { key: "flag", header: "Flagged", align: "right", cell: (r) => (r.detected_count === null ? "–" : `${int(r.detected_count)} ${r.count_unit}`) },
    { key: "rate", header: "Rate", align: "right", cell: (r) => <b>{pct(r.detected_rate)}</b> },
    { key: "prec", header: "Precision", align: "right", cell: (r) => pct(r.benchmark_precision, 0) },
    { key: "rec", header: "Recall", align: "right", cell: (r) => pct(r.benchmark_recall, 0) },
    { key: "src", header: "Measured on", cell: (r) => <span className="text-xs text-muted">{r.reliability_source ?? "–"}</span> },
    { key: "ev", header: "Evidence", cell: (r) => <Status value={r.evidence_status} /> },
    { key: "note", header: "Note", cell: (r) => <span className="block max-w-[22rem] text-xs text-muted">{r.note || "–"}</span> },
  ];
  const chart = rows.filter((r) => r.benchmark_precision !== null && r.benchmark_recall !== null && r.detected_rate !== null)
    .map((r) => ({ name: r.dimension.replace("Potential ", ""), Precision: r.benchmark_precision, Recall: r.benchmark_recall }));
  return (
    <Page title="Quality audit" subtitle="What each detector flagged, and how far to trust the flag. A flagged rate is not a confirmed error rate.">
      <Card pad={false}><div className="p-2"><DataTable columns={cols} rows={rows} rowKey={(r) => r.issue_key} /></div></Card>
      {chart.length > 0 && (
        <Card title="Detector reliability" description={`Precision and recall come from controlled experiments with known injected errors: ${rows.find((r) => r.reliability_source)?.reliability_source ?? "benchmark"}.`}>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chart} margin={{ top: 8, right: 8, left: -8, bottom: 0 }}>
                <CartesianGrid {...gridProps} />
                <XAxis dataKey="name" tick={axisTick} axisLine={axisLine} tickLine={false} />
                <YAxis domain={[0, 1]} tick={axisTick} axisLine={false} tickLine={false} tickFormatter={(v) => `${Math.round(v * 100)}%`} />
                <Tooltip cursor={{ fill: "var(--surface-2)" }} content={<ChartTip fmt={(v) => pct(v, 0)} />} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Bar isAnimationActive={false} dataKey="Precision" fill="var(--c-series-a)" radius={[4, 4, 0, 0]} />
                <Bar isAnimationActive={false} dataKey="Recall" fill="var(--c-series-b)" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <p className="mt-2 text-xs text-muted">Exact checks (missing values, duplicates) are 100% by construction. Statistical detectors trade precision for recall: the label detector is usually right when it fires but finds only a fraction of flipped labels.</p>
        </Card>
      )}
    </Page>
  );
}
