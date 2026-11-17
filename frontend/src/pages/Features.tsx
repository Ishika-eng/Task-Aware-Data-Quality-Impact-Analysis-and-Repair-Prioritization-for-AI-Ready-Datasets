import { Download } from "lucide-react";
import { useMemo, useState } from "react";
import { api, type FeatureRow } from "../api";
import { useAudit } from "../audit-context";
import { Alert, Badge, ButtonLink, Card, DataTable, EmptyState, Page, Segmented, Stat, type Col } from "../components/ui";
import { num, type Tone } from "../lib";

const LABEL: Record<FeatureRow["verdict"], string> = {
  relevant: "Relevant",
  constant: "Constant",
  id_like: "Identifier-like",
  redundant: "Redundant",
  no_detectable_relevance: "No detectable relevance",
  not_assessed: "Not assessed",
};
const TONE: Record<FeatureRow["verdict"], Tone> = {
  relevant: "good",
  constant: "warn",
  id_like: "warn",
  redundant: "warn",
  no_detectable_relevance: "accent",
  not_assessed: "neutral",
};
const SORT: Record<FeatureRow["verdict"], number> = { constant: 0, id_like: 1, redundant: 2, no_detectable_relevance: 3, not_assessed: 4, relevant: 5 };

type View = "flagged" | "all";

export function Features() {
  const a = useAudit();
  const [view, setView] = useState<View>("flagged");
  const rows = a.features;

  const sorted = useMemo(() => [...(rows ?? [])].sort((x, y) => SORT[x.verdict] - SORT[y.verdict] || x.column.localeCompare(y.column)), [rows]);

  if (!rows)
    return (
      <Page title="Feature relevance" subtitle="Columns that carry no usable signal for the model.">
        <EmptyState title="The feature check did not run">
          It could not complete for this audit. The reason, if any, is listed under data notes on the Overview page. Nothing in your cleaned file depends on it.
        </EmptyState>
      </Page>
    );

  const count = (v: FeatureRow["verdict"]) => rows.filter((r) => r.verdict === v).length;
  const drop = rows.filter((r) => r.recommendation.startsWith("drop")).length;
  const shown = view === "flagged" ? sorted.filter((r) => r.verdict !== "relevant") : sorted;
  const noTarget = a.request.target === null || a.request.target === undefined;

  const cols: Col<FeatureRow>[] = [
    { key: "c", header: "Column", cell: (r) => <span className="font-medium">{r.column}</span> },
    { key: "k", header: "Type", cell: (r) => <span className="text-xs text-muted">{r.kind}</span> },
    { key: "v", header: "Verdict", cell: (r) => <Badge tone={TONE[r.verdict]}>{LABEL[r.verdict]}</Badge> },
    { key: "r", header: "Recommendation", cell: (r) => <span className="text-xs">{r.recommendation}</span> },
    {
      key: "e", header: "Importance / probe", align: "right",
      cell: (r) => (r.perm_importance === null ? <span className="text-muted">–</span> : <span className="tabular-nums text-xs">{num(r.perm_importance, 4)} / {num(r.probe_importance_threshold, 4)}</span>),
    },
    { key: "w", header: "Why", cell: (r) => <span className="block max-w-[28rem] text-xs text-muted">{r.reason || "–"}</span> },
  ];

  return (
    <Page
      title="Feature relevance"
      subtitle="Which columns carry no usable signal: constant, identifier-like, redundant, or indistinguishable from random noise. Report-only: nothing is dropped from your cleaned file."
      actions={<ButtonLink size="sm" href={api.downloadUrl(a.id, "feature_report.csv")}><Download className="h-3.5 w-3.5" /> Download feature_report.csv</ButtonLink>}
    >
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Stat label="Columns checked" value={rows.length} />
        <Stat label="Drop candidates" value={drop} tone={drop ? "warn" : "neutral"} hint="constant, identifier or redundant" />
        <Stat label="No detectable relevance" value={count("no_detectable_relevance")} hint="weak evidence: review" />
        <Stat label="Not assessed" value={count("not_assessed")} hint="see the reason column" />
      </div>

      {noTarget && <Alert tone="accent" title="No target column">Only the structural checks ran (constant, identifier-like, redundant). Whether a column predicts the target needs a target to test against.</Alert>}
      <Alert tone="warn" title="A recommendation, not a verdict">
        “No detectable relevance” means the column was no better than random probe columns for a random forest. A feature can still matter only in combination with others, for a different model, or for a different task. Check before removing anything. Columns flagged redundant name the earlier column they duplicate; that one is kept.
      </Alert>

      <Card
        title="Columns"
        description={`Importance is the drop in ${a.overview.task === "regression" ? "R-squared" : "balanced accuracy"} when the column is shuffled on held-out rows; probe is the largest value among 12 random probe columns. A column is flagged only if both its importance and its mutual information with the target are at or below the probes.`}
        actions={<Segmented label="Show" value={view} onChange={setView} options={[{ value: "flagged", label: "Needs attention" }, { value: "all", label: "All columns" }]} />}
      >
        <DataTable columns={cols} rows={shown} rowKey={(r) => r.column} dense empty="Every column looks relevant." />
      </Card>
    </Page>
  );
}
