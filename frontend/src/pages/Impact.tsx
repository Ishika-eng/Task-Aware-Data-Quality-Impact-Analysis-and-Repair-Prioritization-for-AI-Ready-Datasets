import { Bar, BarChart, CartesianGrid, Cell, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { ImpactRow } from "../api";
import { useAudit } from "../audit-context";
import { axisLine, axisTick, ChartTip, gridProps } from "../components/charts";
import { Alert, Badge, Card, DataTable, Dot, Page, Status, type Col } from "../components/ui";
import { ISSUE_COLOR, ISSUE_NAME, pct, signed } from "../lib";

export function Impact() {
  const a = useAudit();
  const rows = a.impact;
  const hasAuc = rows.some((r) => r.impact_estimate_auc !== null);
  const unit = rows.find((r) => r.impact_unit !== "-")?.impact_unit ?? "F1";
  const data = rows.filter((r) => r.benchmark_impact_estimate !== null).map((r) => ({
    key: r.issue_key, name: ISSUE_NAME[r.issue_key] ?? r.issue, F1: r.benchmark_impact_estimate, AUC: r.impact_estimate_auc, boundary: r.is_boundary_estimate,
  }));
  const cols: Col<ImpactRow>[] = [
    { key: "i", header: "Issue", cell: (r) => <span className="flex items-center gap-2 font-medium"><Dot color={ISSUE_COLOR[r.issue_key] ?? "var(--muted)"} />{r.issue}</span> },
    { key: "r", header: "Detected", align: "right", cell: (r) => pct(r.detected_rate) },
    { key: "e", header: `Impact (${unit.replace(" (positive class)", "")})`, align: "right", cell: (r) => r.benchmark_impact_estimate === null ? "–" : <span><b>{signed(r.benchmark_impact_estimate)}</b>{r.is_boundary_estimate && <Badge tone="warn" className="ml-1.5">boundary</Badge>}</span> },
    ...(hasAuc ? [{ key: "a", header: "Impact (ROC-AUC)", align: "right" as const, cell: (r: ImpactRow) => (r.impact_estimate_auc === null ? "–" : signed(r.impact_estimate_auc)) }] : []),
    { key: "d", header: "Metrics", cell: (r) => (r.metrics_disagree ? <Badge tone="warn" className="whitespace-normal">F1 and AUC disagree</Badge> : <span className="text-muted">–</span>) },
    { key: "s", header: "Based on", cell: (r) => <span className="text-xs text-muted">{r.impact_source}</span> },
    { key: "ev", header: "Repair evidence", cell: (r) => <Status value={r.repairability_status} /> },
    { key: "dec", header: "Decision", cell: (r) => <Badge tone={r.policy_decision.startsWith("applied") ? "good" : r.policy_decision.startsWith("flagged") ? "warn" : "neutral"}>{r.policy_decision.replace(" (not applied)", "")}</Badge> },
    { key: "rec", header: "Recommendation", cell: (r) => <span className="block max-w-[24rem] text-xs text-muted">{r.recommendation}</span> },
  ];
  return (
    <Page title="Task impact" subtitle="How much each detected problem is estimated to cost a model. The detected rate is mapped onto a damage curve, then shown with the repair evidence.">
      <Alert tone="warn" title="Benchmark-based estimate, not a prediction">{a.impact_note}</Alert>
      <Card title="Estimated damage per issue" description="Larger is worse. Hatched values are boundary estimates: the detected rate is above the highest measured rate (20%), so the 20% value is shown, never extrapolated.">
        <div className="h-72">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
              <CartesianGrid {...gridProps} />
              <XAxis dataKey="name" tick={axisTick} axisLine={axisLine} tickLine={false} />
              <YAxis tick={axisTick} axisLine={false} tickLine={false} tickFormatter={(v) => v.toFixed(3)} />
              <Tooltip cursor={{ fill: "var(--surface-2)" }} content={<ChartTip fmt={(v) => signed(v)} />} />
              {hasAuc && <Legend wrapperStyle={{ fontSize: 12 }} />}
              <Bar isAnimationActive={false} dataKey="F1" name={unit.replace(" (positive class)", "")} radius={[4, 4, 0, 0]}>
                {data.map((d) => <Cell key={d.key} fill={ISSUE_COLOR[d.key]} fillOpacity={d.boundary ? 0.55 : 1} />)}
              </Bar>
              {hasAuc && <Bar isAnimationActive={false} dataKey="AUC" name="ROC-AUC" fill="var(--muted)" fillOpacity={0.55} radius={[4, 4, 0, 0]} />}
            </BarChart>
          </ResponsiveContainer>
        </div>
      </Card>
      <Card pad={false}><div className="p-2"><DataTable columns={cols} rows={rows} rowKey={(r) => r.issue_key} /></div></Card>
      {rows.filter((r) => r.metrics_disagree).map((r) => <Alert key={r.issue_key} tone="warn" title={`${r.issue}: the two metrics disagree`}>{r.metric_note}</Alert>)}
    </Page>
  );
}
