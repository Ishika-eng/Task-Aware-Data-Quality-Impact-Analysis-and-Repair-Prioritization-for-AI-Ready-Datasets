import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { EvidenceRow } from "../api";
import { useAudit } from "../audit-context";
import { axisLine, axisTick, ChartTip, gridProps } from "../components/charts";
import { Alert, Badge, Button, Card, DataTable, Dot, EmptyState, Page, Segmented, Stat, Status, type Col } from "../components/ui";
import { ISSUE_COLOR, issueName, int, num, pct, seconds, signed } from "../lib";

export function Calibration() {
  const a = useAudit();
  const cal = a.calibration;
  const [metric, setMetric] = useState<"f1" | "auc">("f1");

  const curve = useMemo(() => {
    if (cal.status !== "ok" || !cal.impact_curve) return { rows: [], kinds: [] as string[] };
    const kinds = Array.from(new Set(cal.impact_curve.map((c) => c.error_type)));
    const rates = Array.from(new Set(cal.impact_curve.map((c) => c.rate))).sort((x, y) => x - y);
    const rows = rates.map((rate) => {
      const row: Record<string, number | string | null> = { rate: `${Math.round(rate * 100)}%` };
      for (const k of kinds) {
        const c = cal.impact_curve!.find((x) => x.error_type === k && x.rate === rate);
        row[k] = c ? (metric === "f1" ? c.mean_damage : c.mean_damage_auc ?? null) : null;
      }
      return row;
    });
    return { rows, kinds };
  }, [cal, metric]);

  if (cal.status === "not_run")
    return (
      <Page title="Calibration" subtitle="Measure damage and repair safety on your own data.">
        <EmptyState title="This audit was not calibrated" action={<Link to="/"><Button variant="primary">Run a calibrated audit</Button></Link>}>
          Every decision currently uses the UCI Adult benchmark evidence. Tick “Calibrate on this dataset” (it needs a target column) when you run an audit to replace it with measurements from your own data.
        </EmptyState>
      </Page>
    );
  if (cal.status === "declined")
    return (
      <Page title="Calibration" subtitle="Measure damage and repair safety on your own data.">
        <Alert tone="warn" title="Calibration declined to run">{cal.reason}. Decisions keep using the UCI Adult benchmark evidence. Declining is deliberate: evidence that cannot be meaningful would only look reassuring.</Alert>
        {cal.reference && (
          <Card title="Reference build">
            <div className="grid gap-3 sm:grid-cols-3">
              <Stat label="Labeled rows" value={int(cal.reference.rows_labeled)} />
              <Stat label="Complete rows" value={int(cal.reference.rows_complete)} />
              <Stat label="After de-duplication" value={int(cal.reference.rows_after_dedup)} />
            </div>
          </Card>
        )}
      </Page>
    );

  const ref = cal.reference!;
  const reg = cal.task === "regression";
  const floor = metric === "f1" ? cal.noise_floor! : (cal.noise_floor_auc ?? cal.noise_floor!);
  const funnel = [
    ["Labeled rows", ref.rows_labeled], ["Complete rows", ref.rows_complete], ["Without duplicates", ref.rows_after_dedup], ["Used as reference", ref.rows_used],
  ] as const;
  const ev = cal.evidence ?? [];
  const cols: Col<EvidenceRow>[] = [
    { key: "e", header: "Error", cell: (r) => <span className="flex items-center gap-2 font-medium"><Dot color={ISSUE_COLOR[r.error_type]} />{issueName(r.error_type, cal.task)}</span> },
    { key: "r", header: "Rate", align: "right", cell: (r) => pct(r.rate, 0) },
    { key: "d1", header: reg ? "Damage (R²)" : "Damage (F1)", align: "right", cell: (r) => signed(r.mean_damage) },
    { key: "d2", header: "Damage (AUC)", align: "right", cell: (r) => signed(r.mean_damage_auc) },
    { key: "p", header: "Repair precision", align: "right", cell: (r) => pct(r.repair_precision, 0) },
    { key: "c", header: "Repair recall", align: "right", cell: (r) => pct(r.repair_recall, 0) },
    { key: "rec", header: "Recovery", align: "right", cell: (r) => signed(r.mean_recovery) },
    { key: "s", header: reg ? "Evidence (R²)" : "Evidence F1 / AUC", cell: (r) => <span className="flex gap-1"><Status value={r.status_macrof1} />{!reg && <Status value={r.status_auc} />}</span> },
    { key: "m", header: "Harm risk", align: "right", cell: (r) => num(r.material_risk) },
    { key: "ok", header: "Certifiable", cell: (r) => (r.underpowered ? <Badge tone="warn">no (noisy)</Badge> : <Badge tone="good">yes</Badge>) },
  ];

  return (
    <Page title="Calibration" subtitle={`Measured on a clean-ish reference built from your own data: ${cal.n_seeds} seeds, ${cal.rates?.map((r) => `${Math.round(r * 100)}%`).join(" / ")} corruption, ${cal.metric}. Took ${seconds(cal.seconds ?? 0)}.`}>
      <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-5">
        <Stat label="Reference rows" value={int(ref.rows_used)} hint={`${int(ref.rows_train)} train · ${int(ref.rows_test)} test`} />
        <Stat label={reg ? "Clean R²" : "Clean macro-F1"} value={num(cal.clean_macro_f1, 3)} />
        <Stat label="Chance level" value={num(cal.chance_f1, 3)} hint="same model, shuffled labels" />
        <Stat label={reg ? "Noise floor (R²)" : "Noise floor (F1)"} value={num(cal.noise_floor)} tone={cal.noise_floor! > 0.01 ? "warn" : "neutral"} />
        {cal.has_auc && <Stat label="Noise floor (AUC)" value={num(cal.noise_floor_auc)} tone={cal.noise_floor_auc! > 0.01 ? "warn" : "neutral"} />}
      </div>

      <div className="space-y-3">
        {cal.imbalance_warning && <Alert tone="accent" title="Imbalanced target">The minority class is only {pct(cal.minority_share, 0)} of your reference. F1-type metrics depend on the decision threshold and can mislead on imbalanced targets, so ROC-AUC (which does not) is measured alongside macro-F1, and a repair counts as safe only if neither metric shows material harm.</Alert>}
        {(cal.metric_disagreement ?? []).map((d) => <Alert key={d.error_type} tone="warn" title={`${issueName(d.error_type, cal.task)}: macro-F1 and ROC-AUC disagree`}>{d.note}</Alert>)}
        {cal.underpowered && <Alert tone="warn" title="Too noisy to certify some repairs">The measurement noise on your data is above 0.01, so harm smaller than that would be invisible. Non-exact repairs (imputation, outlier clipping) cannot be certified safe from this calibration and stay in human review. Exact duplicate removal is unaffected. A larger clean-ish reference lowers the noise; more seeds help only modestly.</Alert>}
        {Object.keys(cal.skipped_issues ?? {}).length > 0 && <Alert tone="warn" title="Not measured">{Object.entries(cal.skipped_issues!).map(([k, v]) => `${k} (${v})`).join("; ")}. These keep using the UCI Adult evidence.</Alert>}
      </div>

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1.6fr)_minmax(0,1fr)]">
        <Card title="Damage measured on your data" description="How much each injected error lowers the score compared with a model trained on the clean reference. The dashed line is the noise floor: smaller effects are indistinguishable from training noise."
          actions={cal.has_auc ? <Segmented label="Metric" value={metric} onChange={setMetric} options={[{ value: "f1", label: "macro-F1" }, { value: "auc", label: "ROC-AUC" }]} /> : undefined}>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={curve.rows} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
                <CartesianGrid {...gridProps} />
                <XAxis dataKey="rate" tick={axisTick} axisLine={axisLine} tickLine={false} />
                <YAxis tick={axisTick} axisLine={false} tickLine={false} tickFormatter={(v) => v.toFixed(3)} />
                <Tooltip content={<ChartTip fmt={(v) => signed(v)} />} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <ReferenceLine y={floor} stroke="var(--muted)" strokeDasharray="4 4" />
                <ReferenceLine y={0} stroke="var(--line)" />
                {curve.kinds.map((k) => <Line isAnimationActive={false} key={k} type="monotone" dataKey={k} name={issueName(k, cal.task)} stroke={ISSUE_COLOR[k]} strokeWidth={2} dot={{ r: 3 }} connectNulls />)}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card title="How the reference was built" description="Calibration needs clean data, so it derives the cleanest data it can from yours.">
          <div className="space-y-3">
            {funnel.map(([label, value]) => (
              <div key={label}>
                <div className="mb-1 flex justify-between text-xs"><span>{label}</span><span className="tabular-nums text-muted">{int(value)}</span></div>
                <div className="h-2 rounded-full bg-surface-2"><div className="h-full rounded-full bg-accent" style={{ width: `${(value / ref.rows_labeled) * 100}%` }} /></div>
              </div>
            ))}
          </div>
          <p className="mt-3 text-xs text-muted">{int(ref.rows_removed_label_problems)} rows with high-confidence label problems were also removed. The test half is never corrupted.</p>
          {Object.keys(ref.columns_left_out).length > 0 && <p className="mt-2 text-xs text-muted">Left out: {Object.entries(ref.columns_left_out).map(([c, w]) => `${c} (${w})`).join(", ")}.</p>}
        </Card>
      </div>

      <Card title="Damage and repair evidence" description={`Means over ${cal.n_seeds} seeds. Recovery is how much of the damage the real TaskClean repair wins back. Harm risk is the worse of the two metrics and counts only harm larger than that metric's noise floor.`} pad={false}>
        <div className="p-2"><DataTable columns={cols} rows={ev} rowKey={(r) => `${r.error_type}-${r.rate}`} dense /></div>
      </Card>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card title="Does each detector's flagged rate track real damage on your data?" description="Three rates only: directional evidence, never validated.">
          <div className="space-y-3">
            {(cal.detector_tracking ?? []).map((t) => (
              <div key={t.error_type} className="rounded-lg border border-line p-3">
                <div className="flex items-center justify-between gap-2"><span className="flex items-center gap-2 text-sm font-medium"><Dot color={ISSUE_COLOR[t.error_type]} />{issueName(t.error_type, cal.task)}</span><Status value={t.evidence_status} /></div>
                <p className="mt-1.5 text-xs text-muted">{t.note}</p>
              </div>
            ))}
          </div>
        </Card>
        <Card title="Detector false-positive floors" description="What each detector flags on data that is clean-ish by construction. An observed rate close to this is not evidence of a problem.">
          <div className="space-y-3">
            {Object.entries(cal.baseline_floors ?? {}).map(([k, v]) => (
              <div key={k}>
                <div className="mb-1 flex justify-between text-xs"><span className="flex items-center gap-2"><Dot color={ISSUE_COLOR[k]} />{issueName(k, cal.task)}</span><span className="tabular-nums text-muted">{pct(v)}</span></div>
                <div className="h-2 rounded-full bg-surface-2"><div className="h-full rounded-full" style={{ width: `${Math.min(100, v * 100)}%`, background: ISSUE_COLOR[k] }} /></div>
              </div>
            ))}
          </div>
        </Card>
      </div>
    </Page>
  );
}
