import { Lock } from "lucide-react";
import type { HealthColumn, HealthSnapshot, TargetProfile } from "../api";
import { Alert, Badge, Card, DataTable, Stat, type Col } from "./ui";
import { cellText, int, num, pct } from "../lib";

/** One row of the "problems found" table. `source` says whether the number is the audit's own definition (so the audit will
 *  agree with it) or a report-only check that the audit does not make and that has not been validated. */
interface Problem {
  key: string;
  problem: string;
  count: number | null;
  share: number | null;
  unit: string;
  source: "audit" | "report-only";
  detail: string;
}

const RULE_NAMES: Record<string, string> = {
  format_variant: "case/spacing variants",
  abbreviation: "likely abbreviations",
  ambiguous_abbreviation: "ambiguous abbreviations",
  possible_typo: "possible typos",
};

/** "1 column", "2 columns": the unit strings above are plural. */
const unitFor = (n: number | null, unit: string) => (n === 1 && unit.endsWith("s") ? unit.slice(0, -1) : unit);

const topColumns = (m: Record<string, number>, n = 3) =>
  Object.entries(m).sort((a, b) => b[1] - a[1]).slice(0, n).map(([c, k]) => `${c} (${int(k)})`).join(", ");

function problems(s: HealthSnapshot): Problem[] {
  const c = s.completeness;
  const a = s.anomalies;
  const out: Problem[] = [
    { key: "miss", problem: "Missing cells", count: c.missing_cells, share: c.missing_pct, unit: "cells", source: "audit",
      detail: `${int(c.blank_cells)} blank and ${int(c.placeholder_cells)} placeholder cells (such as ?, N/A, -). ` +
        topColumns(Object.fromEntries(s.columns.filter((x) => x.missing > 0).map((x) => [x.name, x.missing]))) },
    { key: "missrows", problem: "Rows with at least one missing cell", count: c.rows_with_missing, share: c.rows_with_missing_pct, unit: "rows", source: "audit",
      detail: c.columns_entirely_missing.length ? `Entirely empty columns: ${c.columns_entirely_missing.join(", ")}` : "" },
  ];
  if (s.duplicates) {
    out.push({ key: "dup", problem: "Exact duplicate rows", count: s.duplicates.exact_rows, share: s.duplicates.exact_pct, unit: "rows", source: "audit",
      detail: s.duplicates.exact_examples.length ? `First occurrence is kept. Example rows (0-based): ${s.duplicates.exact_examples.join(", ")}` : "" });
    out.push({ key: "near", problem: "Near-duplicate rows (beyond the exact ones)", count: s.duplicates.near_extra_rows, share: s.duplicates.near_extra_pct, unit: "rows", source: "report-only",
      detail: s.duplicates.near_definition });
  }
  if (a.iqr)
    out.push({ key: "iqr", problem: "Rows with an outlier (1.5 × IQR)", count: a.iqr.outlier_rows, share: a.iqr.outlier_pct, unit: "rows", source: "audit",
      detail: `${a.iqr.rule}. Most flagged: ${topColumns(a.iqr.per_column) || "none"}` });
  if (a.zscore)
    out.push({ key: "z", problem: "Rows with an outlier (|Z| > 3)", count: a.zscore.outlier_rows, share: a.zscore.outlier_pct, unit: "rows", source: "report-only",
      detail: `Most flagged: ${topColumns(a.zscore.per_column) || "none"}` });
  if (s.inconsistencies) {
    const inc = s.inconsistencies;
    const rules = Object.entries(inc.by_rule).sort((a, b) => b[1] - a[1]).map(([r, k]) => `${int(k)} ${RULE_NAMES[r] ?? r}`).join(", ");
    const check = (inc.by_rule.abbreviation ?? 0) > 0 ? " “Likely abbreviation” matches are a guess: confirm they are not real categories before repairing." : "";
    out.push({ key: "inc", problem: "Inconsistent spellings", count: inc.cells, share: inc.rows_pct, unit: "cells", source: "audit",
      detail: (rules ? `${rules}.${check} ` : "") + inc.by_column.slice(0, 3).map((b) => `${b.column}: ${b.examples.slice(0, 2).map((e) => `“${e.value}”${e.canonical ? ` → “${e.canonical}”` : ""}`).join(", ")}`).join(" · ") });
  }
  out.push({ key: "skew", problem: "Heavily skewed numeric columns (|skew| > 1.5)", count: a.skewed_columns.length, share: null, unit: "columns", source: "report-only",
    detail: a.skewed_columns.map((x) => `${x.column} (${x.skew.toFixed(1)})`).join(", ") + (a.skewed_columns.length ? ". Distorts linear and neural models, not tree models." : "") });
  out.push({ key: "const", problem: "Constant columns", count: a.constant_columns.length, share: null, unit: "columns", source: "report-only", detail: a.constant_columns.join(", ") });
  out.push({ key: "id", problem: "Possible identifier columns", count: a.possible_identifier_columns.length, share: null, unit: "columns", source: "report-only",
    detail: a.possible_identifier_columns.join(", ") });
  const fmt = s.columns.filter((x) => x.kind === "formatted number stored as text").map((x) => x.name);
  out.push({ key: "fmt", problem: "Formatted numbers stored as text ($1,200, 12%)", count: fmt.length, share: null, unit: "columns", source: "report-only",
    detail: fmt.length ? `${fmt.join(", ")}. The audit reads these as categories, not numbers.` : "" });
  return out;
}

function ProblemsTable({ s }: { s: HealthSnapshot }) {
  const cols: Col<Problem>[] = [
    { key: "p", header: "Check", cell: (r) => <span className="font-medium">{r.problem}</span> },
    { key: "n", header: "Found", align: "right", cell: (r) => <b className={r.count ? "" : "text-muted"}>{int(r.count)} <span className="text-xs font-normal text-muted">{unitFor(r.count, r.unit)}</span></b> },
    { key: "s", header: "Share", align: "right", cell: (r) => (r.share === null ? "–" : pct(r.share, 2)) },
    { key: "src", header: "Definition", cell: (r) => <Badge tone={r.source === "audit" ? "good" : "neutral"}>{r.source === "audit" ? "same as the audit" : "report-only"}</Badge> },
    { key: "d", header: "Detail", cell: (r) => <span className="block max-w-[34rem] text-xs text-muted">{r.detail || "–"}</span> },
  ];
  return <DataTable columns={cols} rows={problems(s)} rowKey={(r) => r.key} dense />;
}

function columnSummary(c: HealthColumn): string {
  if (c.median !== undefined && c.median !== null)
    return `${num(c.min, 2)} · ${num(c.median, 2)} · ${num(c.max, 2)}`;
  if (c.top_values?.length) return c.top_values.map((t) => `${t.value} ${pct(t.share, 0)}`).join(", ");
  return c.examples.slice(0, 3).join(", ") || "–";
}

function ColumnsTable({ s }: { s: HealthSnapshot }) {
  const cols: Col<HealthColumn>[] = [
    { key: "n", header: "Column", cell: (r) => <span className="font-medium">{r.name}</span> },
    { key: "k", header: "Type", cell: (r) => <Badge tone={r.kind.includes("text") && r.kind !== "text / high-cardinality" && r.kind !== "categorical" ? "warn" : "neutral"}>{r.kind}</Badge> },
    { key: "u", header: "Distinct", align: "right", cell: (r) => int(r.n_unique) },
    { key: "m", header: "Missing", align: "right", cell: (r) => (r.missing ? <span>{int(r.missing)} <span className="text-xs text-muted">({pct(r.missing_pct, 1)})</span></span> : <span className="text-muted">0</span>) },
    { key: "s", header: "Min · median · max, or most common", cell: (r) => <span className="text-xs tabular-nums text-muted">{columnSummary(r)}</span> },
    { key: "f", header: "Flags", cell: (r) => <span className="block max-w-[26rem] text-xs text-warn">{r.flags.join(" · ")}</span> },
  ];
  return <DataTable columns={cols} rows={s.columns} rowKey={(r) => r.name} dense />;
}

function RegressionTargetBlock({ t }: { t: TargetProfile }) {
  const f = (v?: number) => (v === undefined ? "–" : Math.abs(v) >= 1000 ? v.toLocaleString(undefined, { maximumFractionDigits: 0 }) : v.toLocaleString(undefined, { maximumSignificantDigits: 4 }));
  return (
    <Card title={`Target column: ${t.target} (regression)`} description="A numeric target. Computed for the column you chose; it is not part of the stored snapshot, because the target is chosen after upload.">
      <div className="grid gap-3 sm:grid-cols-4">
        <Stat label="Distinct values" value={int(t.n_distinct ?? 0)} />
        <Stat label="Rows without a target" value={int(t.rows_without_target)} hint={pct(t.rows_without_target_pct, 2)} tone={t.rows_without_target ? "warn" : "neutral"} />
        <Stat label="Mean · std" value={`${f(t.mean)} · ${f(t.std)}`} />
        <Stat label="Skewness" value={t.skew === undefined ? "–" : t.skew.toFixed(2)} tone={t.skew !== undefined && Math.abs(t.skew) > 2 ? "warn" : "neutral"} />
      </div>
      <p className="mt-3 text-sm text-muted tabular-nums">Min {f(t.min)} · 25% {f(t.q25)} · median {f(t.median)} · 75% {f(t.q75)} · max {f(t.max)}</p>
      {t.warnings.map((w) => <div key={w} className="mt-3"><Alert tone="warn">{w}</Alert></div>)}
    </Card>
  );
}

export function TargetBlock({ t }: { t: TargetProfile }) {
  if (t.task === "regression") return <RegressionTargetBlock t={t} />;
  return (
    <Card title={`Target column: ${t.target}`} description="Computed for the column you chose. It is not part of the stored snapshot, because the target is chosen after upload.">
      <div className="grid gap-3 sm:grid-cols-4">
        <Stat label="Classes" value={int(t.n_classes)} />
        <Stat label="Rows without a target" value={int(t.rows_without_target)} hint={pct(t.rows_without_target_pct, 2)} tone={t.rows_without_target ? "warn" : "neutral"} />
        <Stat label="Imbalance" value={t.imbalance_ratio ? `${t.imbalance_ratio.toFixed(1)} : 1` : "–"} hint="largest class to smallest" />
        <Stat label="Smallest class" value={t.minority_share !== undefined ? pct(t.minority_share, 1) : "–"} />
      </div>
      {t.warnings.map((w) => <div key={w} className="mt-3"><Alert tone="warn">{w}</Alert></div>)}
      <div className="mt-4 space-y-1.5">
        {t.classes.slice(0, 12).map((c) => (
          <div key={c.value} className="flex items-center gap-3 text-sm">
            <span className="w-40 truncate">{cellText(c.value)}</span>
            <div className="h-2 flex-1 overflow-hidden rounded-full bg-surface-2"><div className="h-full rounded-full bg-accent" style={{ width: `${Math.max(1, c.share * 100)}%` }} /></div>
            <span className="w-28 text-right text-xs tabular-nums text-muted">{int(c.rows)} · {pct(c.share, 1)}</span>
          </div>
        ))}
      </div>
    </Card>
  );
}

export function HealthReport({ snapshot: s, target, actions, auditTarget }: { snapshot: HealthSnapshot; target?: TargetProfile | null; actions?: React.ReactNode; auditTarget?: string | null }) {
  const ov = s.overview;
  const types = Object.entries(ov.type_counts).sort((a, b) => b[1] - a[1]);
  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2 rounded-xl border border-line bg-surface-2/60 px-4 py-3 text-xs text-muted">
        <span className="inline-flex items-center gap-1.5 font-medium text-fg"><Lock className="h-3.5 w-3.5" /> Captured on upload, before any cleaning. It never changes.</span>
        <span>{new Date(s.captured_at_utc).toLocaleString()}</span>
        {ov.sha256 && <span title={ov.sha256}>SHA-256 <code className="font-mono">{ov.sha256.slice(0, 16)}…</code></span>}
        {actions && <span className="ml-auto flex gap-2">{actions}</span>}
      </div>

      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <Stat label="Rows" value={int(ov.rows)} />
        <Stat label="Columns" value={int(ov.columns)} />
        <Stat label="File size" value={ov.file_mb === null ? "–" : `${ov.file_mb.toFixed(2)} MB`} hint={`${ov.memory_mb.toFixed(2)} MB in memory`} />
        <Stat label="Missing cells" value={pct(s.completeness.missing_pct, 2)} tone={s.completeness.missing_cells ? "warn" : "good"} hint={`${int(s.completeness.missing_cells)} of ${int(s.completeness.total_cells)}`} />
        <Stat label="Exact duplicates" value={s.duplicates ? int(s.duplicates.exact_rows) : "–"} tone={s.duplicates?.exact_rows ? "warn" : "good"} hint={s.duplicates ? `${pct(s.duplicates.exact_pct, 2)} of rows` : undefined} />
        <Stat label="Outlier rows (IQR)" value={s.anomalies.iqr ? pct(s.anomalies.iqr.outlier_pct, 1) : "–"} hint={s.anomalies.iqr ? `${int(s.anomalies.iqr.outlier_rows)} rows` : undefined} />
        <Stat label="Inconsistent spellings" value={s.inconsistencies ? int(s.inconsistencies.cells) : "–"} hint="cells" tone={s.inconsistencies?.cells ? "warn" : "good"} />
      </div>

      <Card title="What the file is" description={`${ov.file_name || "uploaded file"} · ${ov.input_format.format && ov.input_format.format !== "csv" ? `read as a ${String(ov.input_format.format).toUpperCase()} file` : `read as ${String(ov.input_format.encoding ?? "?")}, “${String(ov.input_format.delimiter ?? "?")}” delimiter, decimal “${String(ov.input_format.decimal ?? ".")}”`}`}>
        <div className="flex flex-wrap gap-2">
          {types.map(([k, n]) => <Badge key={k} tone={k.includes("stored as text") ? "warn" : "neutral"}>{n} × {k}</Badge>)}
        </div>
        {s.notes.length > 0 && <ul className="mt-3 list-disc space-y-1 pl-5 text-xs text-muted">{s.notes.map((n) => <li key={n}>{n}</li>)}</ul>}
      </Card>

      <Card title="What is wrong with it" description="Counts and rates, not a score. “Same as the audit” means the audit uses the same definition and will report the same number; “report-only” checks are not validated and nothing is repaired from them." pad={false}>
        <div className="p-2"><ProblemsTable s={s} /></div>
        {auditTarget && (
          <p className="border-t border-line px-5 py-3 text-xs leading-relaxed text-muted">
            This snapshot is taken at upload, before a target is chosen, so it treats <b className="text-fg">{auditTarget}</b> like any other column. The audit leaves the target out of its outlier check, so “same as the audit” counts can differ slightly (for example outlier rows) when the target is numeric.
          </p>
        )}
      </Card>

      {target && <TargetBlock t={target} />}

      <Card title="Column by column" pad={false}><div className="p-2"><ColumnsTable s={s} /></div></Card>

      <Alert tone="accent" title="What this report does not cover">
        <ul className="list-disc space-y-1 pl-5">{s.not_covered.map((n) => <li key={n}>{n}</li>)}</ul>
      </Alert>
    </div>
  );
}
