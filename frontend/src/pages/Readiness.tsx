import type { ReadinessRow } from "../api";
import { useAudit } from "../audit-context";
import { Alert, Badge, Card, DataTable, Disclosure, Dot, Page, RateBar, Status, type Col } from "../components/ui";
import { ISSUE_COLOR, pct, tone } from "../lib";

export function Readiness() {
  const a = useAudit();
  const cols: Col<ReadinessRow>[] = [
    { key: "d", header: "Dimension", cell: (r) => <span className="flex items-center gap-2 font-medium"><Dot color={ISSUE_COLOR[r.issue_key] ?? "var(--muted)"} />{r.dimension}</span> },
    { key: "o", header: "Observed", align: "right", cell: (r) => <b>{pct(r.observed_rate)}</b> },
    { key: "bar", header: "vs clean-data floor", cell: (r) => <div className="w-40"><RateBar value={r.observed_rate} floor={r.baseline_false_positive_floor} color={ISSUE_COLOR[r.issue_key] ?? "var(--muted)"} /></div> },
    { key: "f", header: "Floor", align: "right", cell: (r) => pct(r.baseline_false_positive_floor) },
    { key: "after", header: "After cleaning", align: "right", cell: (r) => pct(r.rate_after_cleaning) },
    { key: "ev", header: "Evidence", cell: (r) => <Status value={r.evidence_status} /> },
    { key: "st", header: "Status", cell: (r) => <Badge tone={tone(r.status)}>{r.status}</Badge> },
  ];
  return (
    <Page title="Readiness" subtitle="A multidimensional report. Each dimension carries its own evidence status; nothing is collapsed into one number.">
      <Alert tone="accent" title="Why there is no single readiness score">
        Across 15 (error type, rate) points the correlation between a detector's flagged rate and the damage actually measured was r = −0.24 (p = 0.39). A score built from those rates would not be trustworthy, so the evidence is shown per dimension instead.
      </Alert>
      <Card pad={false}><div className="p-2"><DataTable columns={cols} rows={a.readiness} rowKey={(r) => r.issue_key} /></div></Card>
      <p className="-mt-2 text-xs text-muted">
        “After cleaning” is recomputed only for the deterministic detectors (missing values, duplicates, outliers, consistency); model-based detectors are not re-run. The floor is what each detector flags on data that is clean (UCI Adult, or your reference if you calibrated): a rate near the floor is not evidence of a problem.
      </p>
      <Card title="Evidence notes per dimension">
        <div className="space-y-2">
          {a.readiness.map((r) => (
            <Disclosure key={r.issue_key} title={<span className="flex flex-wrap items-center gap-2">{r.dimension} <Status value={r.evidence_status} /></span>}>
              <p>{r.evidence_note || "No additional note."}</p>
              {r.baseline_source && <p className="mt-2 text-xs">Floor source: {r.baseline_source}.</p>}
            </Disclosure>
          ))}
        </div>
      </Card>
      <Card title="Limitations" description="These switch with the calibration state.">
        <ul className="list-disc space-y-2 pl-5 text-sm text-muted">{a.limitations.map((l) => <li key={l}>{l}</li>)}</ul>
      </Card>
    </Page>
  );
}
