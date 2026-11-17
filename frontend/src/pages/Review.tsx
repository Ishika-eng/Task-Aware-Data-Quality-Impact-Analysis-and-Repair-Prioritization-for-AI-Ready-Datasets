import { useQuery } from "@tanstack/react-query";
import { ChevronLeft, ChevronRight, Download } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { api, type RepairLogRow } from "../api";
import { useAudit } from "../audit-context";
import { Alert, Badge, Button, ButtonLink, Card, DataTable, Dot, EmptyState, Page, Segmented, Stat, Status, type Col } from "../components/ui";
import { ISSUE_COLOR, issueName, cellText, int, num, pct } from "../lib";

type Kind = "all" | "flagged_for_review" | "flagged_only";

const KIND_LABEL: Record<string, string> = {
  flagged_for_review: "Repair proposed",
  flagged_only: "Detected only",
};

export function Review() {
  const a = useAudit();
  const [variant, setVariant] = useState<"evidence" | "aggressive">("evidence");
  const [kind, setKind] = useState<Kind>("all");
  const [issue, setIssue] = useState("");
  const [page, setPage] = useState(1);
  const size = 25;
  useEffect(() => setPage(1), [variant, kind, issue]);

  const action = kind === "all" ? "flagged_for_review,flagged_only" : kind;
  const q = useQuery({
    queryKey: ["review", a.id, variant, action, issue, page, a.selection.label],
    queryFn: () => api.repairLog(a.id, { variant, action, issue: issue || undefined, page, page_size: size }),
    placeholderData: (prev) => prev,
  });
  const d = q.data;
  const pages = d ? Math.max(1, Math.ceil(d.total / size)) : 1;

  const policy = useMemo(() => Object.fromEntries(a.policy.map((p) => [p.issue_key, p])), [a.policy]);
  const rv = a.review;
  const ev = variant === "evidence"; // summary counts describe the evidence-based file only
  const keyOf = useMemo(() => Object.fromEntries(rv.by_issue.map((r) => [r.problem, r.issue_key])), [rv]);

  if (rv.total === 0)
    return (
      <Page title="Review queue" subtitle="Everything TaskClean flagged but did not change on its own.">
        <EmptyState title="Nothing is waiting for review" action={<Link to={`/audit/${a.id}/outputs`}><Button variant="primary">Go to outputs</Button></Link>}>
          Either no problems were detected, or every proposed repair was applied automatically. The full record is in the repair log on the Outputs page.
        </EmptyState>
      </Page>
    );

  const cols: Col<RepairLogRow>[] = [
    { key: "row", header: "Row", align: "right", cell: (r) => <span className="tabular-nums">{int(r.row_id)}</span> },
    { key: "col", header: "Column", cell: (r) => r.column ?? "–" },
    {
      key: "prob", header: "Problem",
      cell: (r) => {
        const key = keyOf[r.problem];
        return <span className="flex items-center gap-2 whitespace-nowrap"><Dot color={key ? ISSUE_COLOR[key] : "var(--muted)"} />{r.problem}</span>;
      },
    },
    { key: "orig", header: "Current value", cell: (r) => <span className="font-mono text-xs">{cellText(r.original_value)}</span> },
    { key: "prop", header: "Proposed value", cell: (r) => (r.action === "flagged_only" ? <span className="text-muted">none</span> : <span className="font-mono text-xs">{cellText(r.proposed_value)}</span>) },
    { key: "kind", header: "Type", cell: (r) => <Badge tone={r.action === "flagged_only" ? "neutral" : "warn"}>{KIND_LABEL[r.action] ?? r.action}</Badge> },
    { key: "score", header: "Score", align: "right", cell: (r) => (r.detector_score === null ? "–" : num(r.detector_score, 2)) },
    { key: "m", header: "Method", cell: (r) => <span title={r.method} className="block max-w-[16rem] truncate text-xs text-muted">{r.method}</span> },
  ];
  const sel = "rounded-lg border border-line bg-surface px-2.5 py-1.5 text-sm";

  return (
    <Page
      title="Review queue"
      subtitle="Everything TaskClean flagged but did not change on its own. Nothing here is in the cleaned file unless you override it on the Repair plan page."
      actions={<ButtonLink size="sm" href={api.reviewCsvUrl(a.id, variant)}><Download className="h-3.5 w-3.5" /> Download review_queue.csv</ButtonLink>}
    >
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Stat label="Flagged in total" value={int(rv.total)} tone="warn" hint="evidence-based file" />
        <Stat label="Repair proposed" value={int(rv.proposed)} hint="a fix exists, not applied" />
        <Stat label="Detected only" value={int(rv.flag_only)} hint="no safe fix to propose" />
        <Stat label="Issue types" value={rv.by_issue.length} hint="with items to review" />
      </div>

      {a.selection.applied.length === 0 && (
        <Alert tone="accent" title="No repair type was auto-applied">Nothing cleared the safety bar, so every proposal sits in this queue and the evidence-based file equals your upload.</Alert>
      )}

      <div>
        <h2 className="mb-3 text-sm font-semibold">Why each type needs a human</h2>
        <div className="grid gap-4 lg:grid-cols-2">
          {rv.by_issue.map((r) => {
            const p = policy[r.issue_key];
            const never = a.never_auto_apply.includes(r.issue_key);
            return (
              <Card key={r.issue_key}>
                <div className="flex flex-wrap items-center gap-2">
                  <span className="flex items-center gap-2 font-semibold"><Dot color={ISSUE_COLOR[r.issue_key]} />{issueName(r.issue_key, a.overview.task)}</span>
                  {p && <Status value={p.repairability_status} />}
                  {never && <Badge tone="neutral">never automatic</Badge>}
                </div>
                <p className="mt-1 text-sm text-muted"><b className="text-fg">{int(r.proposed)}</b> repair proposed · <b className="text-fg">{int(r.flag_only)}</b> detected only</p>
                {a.held_back?.[r.issue_key] && (
                  <p className="mt-2 rounded-lg border border-warn/40 bg-warn-soft px-3 py-2 text-[13px] leading-relaxed text-fg">
                    <b>{int(a.held_back[r.issue_key].count)}</b> of these are here only because the calibration did not cover their column ({Object.entries(a.held_back[r.issue_key].columns).map(([c, n]) => `${c} ${int(n)}`).join(", ")}). The rule below is met for the columns it did cover, which were applied. A blank in these columns may mean "not applicable" rather than "missing".
                  </p>
                )}
                {p && <p className="mt-2 text-[13px] leading-relaxed text-muted">{p.reason}</p>}
                {p && (
                  <div className="mt-3 flex flex-wrap gap-x-5 gap-y-1 text-xs text-muted">
                    <span>Repair precision <b className="text-fg">{pct(p.benchmark_repair_precision, 0)}</b></span>
                    <span>Observed harm risk <b className="text-fg">{num(p.benchmark_risk)}</b></span>
                    <span>Source: {p.evidence_source}</span>
                    <span>Model check: <Badge tone={p.model_check === "model-dependent" ? "warn" : "neutral"}>{p.model_check}</Badge></span>
                  </div>
                )}
                <div className="mt-3 flex gap-2">
                  <Button size="sm" onClick={() => { setIssue(r.issue_key); document.getElementById("review-table")?.scrollIntoView({ behavior: "smooth" }); }}>Show these rows</Button>
                  <Link to={`/audit/${a.id}/repair`}><Button size="sm" variant="ghost">Override on repair plan</Button></Link>
                </div>
              </Card>
            );
          })}
        </div>
      </div>

      <div id="review-table">
        <Card
          title="Flagged items"
          description="row is the 0-based row position in the uploaded file. A flag is a suspicion, not a confirmed error."
          actions={<Segmented label="File" value={variant} onChange={setVariant} options={[{ value: "evidence", label: "Evidence-based" }, { value: "aggressive", label: "Aggressive" }]} />}
        >
          <div className="mb-3 flex flex-wrap items-center gap-2">
            <select aria-label="Filter by problem" value={issue} onChange={(e) => setIssue(e.target.value)} className={sel}>
              <option value="">All problems</option>
              {rv.by_issue.map((r) => <option key={r.issue_key} value={r.issue_key}>{issueName(r.issue_key, a.overview.task)}{ev ? ` (${int(r.proposed + r.flag_only)})` : ""}</option>)}
            </select>
            <select aria-label="Filter by type" value={kind} onChange={(e) => setKind(e.target.value as Kind)} className={sel}>
              <option value="all">All types</option>
              <option value="flagged_for_review">Repair proposed{ev ? ` (${int(rv.proposed)})` : ""}</option>
              <option value="flagged_only">Detected only{ev ? ` (${int(rv.flag_only)})` : ""}</option>
            </select>
            {(issue || kind !== "all") && <Button size="sm" variant="ghost" onClick={() => { setIssue(""); setKind("all"); }}>Clear filters</Button>}
            <span className="ml-auto text-xs text-muted">{d ? `${int(d.total)} items` : "Loading…"}</span>
          </div>
          <DataTable columns={cols} rows={d?.rows ?? []} rowKey={(r) => r.repair_id} dense empty={q.isLoading ? "Loading…" : "No items match these filters."} />
          <div className="mt-3 flex items-center justify-between text-sm">
            <span className="text-muted">Page {page} of {int(pages)}</span>
            <div className="flex gap-2">
              <Button size="sm" onClick={() => setPage((p) => p - 1)} disabled={page <= 1}><ChevronLeft className="h-3.5 w-3.5" /> Previous</Button>
              <Button size="sm" onClick={() => setPage((p) => p + 1)} disabled={page >= pages}>Next <ChevronRight className="h-3.5 w-3.5" /></Button>
            </div>
          </div>
        </Card>
      </div>
    </Page>
  );
}
