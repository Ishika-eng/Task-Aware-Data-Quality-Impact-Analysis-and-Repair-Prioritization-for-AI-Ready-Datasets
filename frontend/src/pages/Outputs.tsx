import { useQuery } from "@tanstack/react-query";
import { ChevronLeft, ChevronRight, Download, FileText, PackageOpen } from "lucide-react";
import { useEffect, useState } from "react";
import { api, type RepairLogRow } from "../api";
import { useAudit } from "../audit-context";
import { Alert, Badge, ButtonLink, Button, Card, DataTable, Page, Segmented, Stat, Status, type Col } from "../components/ui";
import { issueName, bytes, cellText, int, num, prettyStatus, tone } from "../lib";

const ISSUE_BY_PROBLEM: Record<string, string> = {
  "Missing value": "missing_values", "Duplicate row": "duplicates", "Potential outlier": "outliers",
  "Potential label error": "label_errors", "Potential feature anomaly": "feature_corruption",
  "Inconsistent value": "inconsistency", "Unusual combination of values": "joint_outliers",
};

function RepairLog({ auditId, counts }: { auditId: string; counts: Record<string, number> | undefined }) {
  const [variant, setVariant] = useState<"evidence" | "aggressive">("evidence");
  const [action, setAction] = useState("");
  const [issue, setIssue] = useState("");
  const [page, setPage] = useState(1);
  const size = 25;
  useEffect(() => setPage(1), [variant, action, issue]);
  const q = useQuery({
    queryKey: ["log", auditId, variant, action, issue, page],
    queryFn: () => api.repairLog(auditId, { variant, action: action || undefined, issue: issue || undefined, page, page_size: size }),
    placeholderData: (prev) => prev,
  });
  const d = q.data;
  const pages = d ? Math.max(1, Math.ceil(d.total / size)) : 1;
  const cols: Col<RepairLogRow>[] = [
    { key: "id", header: "Repair", cell: (r) => <span className="font-mono text-xs">{r.repair_id}</span> },
    { key: "row", header: "Row", align: "right", cell: (r) => r.row_id },
    { key: "col", header: "Column", cell: (r) => r.column ?? "–" },
    { key: "prob", header: "Problem", cell: (r) => r.problem },
    { key: "orig", header: "Original", cell: (r) => <span className="font-mono text-xs">{cellText(r.original_value)}</span> },
    { key: "prop", header: "Proposed", cell: (r) => <span className="font-mono text-xs">{cellText(r.proposed_value)}</span> },
    { key: "app", header: "Applied", cell: (r) => <span className="font-mono text-xs">{cellText(r.applied_value)}</span> },
    { key: "act", header: "Action", cell: (r) => <Badge tone={tone(r.action)}>{prettyStatus(r.action)}</Badge> },
    { key: "tier", header: "Tier", cell: (r) => <span className="text-xs text-muted">{r.tier}</span> },
    { key: "ov", header: "Override", cell: (r) => (r.human_override ? <Badge tone="warn">human</Badge> : <span className="text-muted">–</span>) },
    { key: "m", header: "Method", cell: (r) => <span className="block max-w-[18rem] text-xs text-muted">{r.method}</span> },
  ];
  const sel = "rounded-lg border border-line bg-surface px-2.5 py-1.5 text-sm";
  return (
    <Card title="Repair log" description="Every proposal and every non-change. row_id is the 0-based row position in the uploaded file."
      actions={<Segmented label="File" value={variant} onChange={setVariant} options={[{ value: "evidence", label: "Evidence-based file" }, { value: "aggressive", label: "Aggressive file" }]} />}>
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <select aria-label="Filter by action" value={action} onChange={(e) => setAction(e.target.value)} className={sel}>
          <option value="">All actions</option>
          {Object.entries(d?.actions ?? counts ?? {}).map(([k, v]) => <option key={k} value={k}>{prettyStatus(k)} ({int(v)})</option>)}
        </select>
        <select aria-label="Filter by problem" value={issue} onChange={(e) => setIssue(e.target.value)} className={sel}>
          <option value="">All problems</option>
          {Object.keys(d?.problems ?? {}).map((p) => <option key={p} value={ISSUE_BY_PROBLEM[p] ?? ""}>{p} ({int(d!.problems[p])})</option>)}
        </select>
        <span className="ml-auto text-xs text-muted">{d ? `${int(d.total)} rows` : "Loading…"}</span>
      </div>
      <DataTable columns={cols} rows={d?.rows ?? []} rowKey={(r) => r.repair_id} dense empty={q.isLoading ? "Loading…" : "No rows match these filters."} />
      <div className="mt-3 flex items-center justify-between text-sm">
        <span className="text-muted">Page {page} of {int(pages)}</span>
        <div className="flex gap-2">
          <Button size="sm" onClick={() => setPage((p) => p - 1)} disabled={page <= 1}><ChevronLeft className="h-3.5 w-3.5" /> Previous</Button>
          <Button size="sm" onClick={() => setPage((p) => p + 1)} disabled={page >= pages}>Next <ChevronRight className="h-3.5 w-3.5" /></Button>
        </div>
      </div>
    </Card>
  );
}

export function Outputs() {
  const a = useAudit();
  const s = a.summary.evidence_based, g = a.summary.aggressive;
  const nameOf = (n: string) => n.replace(/_/g, " ").replace(".csv", "").replace(".json", "");
  return (
    <Page title="Outputs" subtitle="TaskClean produces two cleaned files. Pick the one that matches how much risk you accept.">
      {s.human_overrides.length > 0 && <Alert tone="warn" title="Human overrides applied to the evidence-based file">{s.human_overrides.map((k) => issueName(k, a.overview.task)).join(", ")}</Alert>}
      <div className="grid gap-6 lg:grid-cols-2">
        <Card title="Evidence-based" description={`dataset_cleaned.csv · repairs applied using: ${a.selection.label}. Only repairs that never showed harm in the evidence; everything else is logged, not changed.`}>
          <div className="grid grid-cols-3 gap-3">
            <Stat label="Rows" value={int(s.rows_after_cleaning)} hint={s.rows_dropped ? `−${int(s.rows_dropped)} dropped` : undefined} />
            <Stat label="Cells modified" value={int(s.cells_modified)} />
            <Stat label="Flagged" value={int(s.flagged_for_review)} tone="warn" hint="for review" />
          </div>
          <div className="mt-4 flex flex-wrap gap-2">
            <ButtonLink variant="primary" size="sm" href={api.downloadUrl(a.id, "dataset_cleaned.csv")}><Download className="h-3.5 w-3.5" /> Cleaned file</ButtonLink>
            <ButtonLink size="sm" href={api.downloadUrl(a.id, "repair_log.csv")}><FileText className="h-3.5 w-3.5" /> Repair log</ButtonLink>
          </div>
          {s.skipped_row_dropped > 0 && <p className="mt-3 text-xs text-muted">{int(s.skipped_row_dropped)} cell repairs were skipped because their row was removed as a duplicate.</p>}
        </Card>
        <Card title="Aggressive" description="dataset_cleaned_aggressive.csv · every proposed repair except feature-anomaly repairs (harmful in the benchmark). Model-ready, but most of these repairs did not clear the safety bar: review the log before trusting it.">
          <div className="grid grid-cols-3 gap-3">
            <Stat label="Rows" value={int(g.rows_after_cleaning)} hint={g.rows_dropped ? `−${int(g.rows_dropped)} dropped` : undefined} />
            <Stat label="Cells modified" value={int(g.cells_modified)} tone="warn" />
            <Stat label="Left flag-only" value={int(g.flagged_only)} />
          </div>
          <div className="mt-4 flex flex-wrap gap-2">
            <ButtonLink size="sm" href={api.downloadUrl(a.id, "dataset_cleaned_aggressive.csv")}><Download className="h-3.5 w-3.5" /> Cleaned file</ButtonLink>
            <ButtonLink size="sm" href={api.downloadUrl(a.id, "repair_log_aggressive.csv")}><FileText className="h-3.5 w-3.5" /> Repair log</ButtonLink>
          </div>
        </Card>
      </div>

      <RepairLog auditId={a.id} counts={undefined} />

      <Card title="All downloads" actions={<ButtonLink variant="primary" size="sm" href={api.zipUrl(a.id)}><PackageOpen className="h-3.5 w-3.5" /> Download everything (.zip)</ButtonLink>}>
        <div className="grid gap-2 sm:grid-cols-2 xl:grid-cols-4">
          {a.files.map((f) => (
            <a key={f.name} href={api.downloadUrl(a.id, f.name)} className="flex items-center justify-between gap-2 rounded-lg border border-line px-3 py-2.5 transition hover:border-accent hover:bg-accent-soft">
              <span className="min-w-0"><span className="block truncate text-sm font-medium">{f.name}</span><span className="text-xs text-muted">{nameOf(f.name)} · {bytes(f.bytes)}</span></span>
              <Download className="h-4 w-4 shrink-0 text-muted" />
            </a>
          ))}
        </div>
        <p className="mt-3 text-xs text-muted">Numeric summary: {int(s.rows_uploaded)} rows uploaded · {num(s.rows_after_cleaning / Math.max(1, s.rows_uploaded) * 100, 1)}% kept in the evidence-based file.</p>
      </Card>
    </Page>
  );
}
