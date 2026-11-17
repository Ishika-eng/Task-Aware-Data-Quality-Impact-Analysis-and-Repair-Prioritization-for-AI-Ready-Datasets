import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Check, Loader2, RotateCcw } from "lucide-react";
import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api } from "../api";
import { useAudit } from "../audit-context";
import { Alert, Badge, Button, Card, Dot, Page, Status, Switch } from "../components/ui";
import { ISSUE_COLOR, issueName, int, num, pct } from "../lib";

export function RepairPlan() {
  const a = useAudit();
  const qc = useQueryClient();
  const nav = useNavigate();
  const key = a.selection.applied.join(",");
  const [sel, setSel] = useState<Set<string>>(new Set(a.selection.applied));
  useEffect(() => setSel(new Set(a.selection.applied)), [key]); // eslint-disable-line react-hooks/exhaustive-deps
  const def = new Set(a.selection.default);
  const same = sel.size === def.size && [...sel].every((k) => def.has(k));
  const unchanged = sel.size === a.selection.applied.length && a.selection.applied.every((k) => sel.has(k));

  const apply = useMutation({
    mutationFn: () => api.apply(a.id, [...sel]),
    onSuccess: (data) => { qc.setQueryData(["audit", a.id], data); nav(`/audit/${a.id}/outputs`); },
  });
  const toggle = (k: string, on: boolean) => setSel((s) => { const n = new Set(s); if (on) n.add(k); else n.delete(k); return n; });

  return (
    <Page title="Repair plan" subtitle="Each issue type is either auto-repaired (zero observed harm and high repair precision in the evidence) or flagged for human review: proposals are logged but nothing is changed. You can override; overrides are recorded in the repair log.">
      {apply.error && <Alert tone="bad" title="Could not apply">{(apply.error as Error).message}</Alert>}
      <div className="space-y-4">
        {a.policy.map((p) => {
          const on = sel.has(p.issue_key);
          const candidates = p.n_repair_candidates;
          const isDefault = def.has(p.issue_key);
          return (
            <Card key={p.issue_key}>
              <div className="flex items-start gap-4">
                <div className="pt-0.5"><Switch label={`Apply ${p.issue}`} checked={on} onChange={(v) => toggle(p.issue_key, v)} disabled={candidates === 0} /></div>
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="flex items-center gap-2 font-semibold"><Dot color={ISSUE_COLOR[p.issue_key]} />{issueName(p.issue_key, a.overview.task)}</span>
                    <Status value={p.default_action} />
                    {a.never_auto_apply.includes(p.issue_key) && <Badge tone="neutral">never automatic</Badge>}
                  </div>
                  <p className="mt-1 text-sm text-muted">
                    <b className="text-fg">{int(candidates)}</b> repair candidate{candidates === 1 ? "" : "s"} · <b className="text-fg">{int(p.n_flag_only)}</b> flag-only
                    {p.detected_rate !== null && <> · detected rate <b className="text-fg">{pct(p.detected_rate)}</b></>}
                  </p>
                  {a.held_back?.[p.issue_key] && (
                    <div className="mt-2"><Alert tone="warn" title={`${int(a.held_back[p.issue_key].count)} of these ${int(candidates)} are held back for review`}>
                      The calibration could not benchmark {Object.keys(a.held_back[p.issue_key].columns).length} column{Object.keys(a.held_back[p.issue_key].columns).length === 1 ? "" : "s"} (too empty to build a clean reference), so its evidence does not cover them: {Object.entries(a.held_back[p.issue_key].columns).map(([c, n]) => `${c} (${int(n)})`).join(", ")}. Only the other {int(candidates - a.held_back[p.issue_key].count)} are applied. A blank in those columns may mean "not applicable" rather than "missing". The held-back items are in the review queue; the aggressive file applies them.
                    </Alert></div>
                  )}
                  <p className="mt-2 text-[13px] leading-relaxed text-muted">{p.reason}</p>
                  <div className="mt-3 flex flex-wrap gap-x-5 gap-y-1 text-xs text-muted">
                    <span>Evidence: <Status value={p.repairability_status} /></span>
                    <span>Repair precision <b className="text-fg">{pct(p.benchmark_repair_precision, 0)}</b></span>
                    <span>Observed harm risk <b className="text-fg">{num(p.benchmark_risk)}</b></span>
                    <span>Source: {p.evidence_source}</span>
                    <span>Model check: <Badge tone={p.model_check === "model-dependent" ? "warn" : "neutral"}>{p.model_check}</Badge></span>
                  </div>
                  {on && p.repairability_status === "negative" && <div className="mt-3"><Alert tone="warn">Benchmark testing found this repair strategy net-harmful. That does not mean the error is unrepairable, only that this strategy made models worse in testing.</Alert></div>}
                  {on && !isDefault && <div className="mt-3"><Alert tone="warn">Human override: the evidence does not support applying this automatically.</Alert></div>}
                </div>
              </div>
            </Card>
          );
        })}
      </div>

      <div className="sticky bottom-4 z-10 flex flex-wrap items-center justify-between gap-3 rounded-xl border border-line bg-surface/95 px-5 py-3 shadow-lg backdrop-blur">
        <div className="text-sm">
          <b>{sel.size}</b> repair type{sel.size === 1 ? "" : "s"} selected · <span className="text-muted">{same ? "matches the policy default" : "custom selection"}</span>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="ghost" onClick={() => setSel(new Set(a.selection.default))} disabled={same}><RotateCcw className="h-4 w-4" /> Reset to policy</Button>
          <Button variant="primary" onClick={() => apply.mutate()} disabled={apply.isPending || unchanged}>
            {apply.isPending ? <Loader2 className="h-4 w-4 animate-spin" /> : <Check className="h-4 w-4" />} Apply selected repairs
          </Button>
        </div>
      </div>
      <p className="text-xs text-muted">Currently applied: <b>{a.selection.label}</b>. See the result on the <Link to={`/audit/${a.id}/outputs`} className="text-accent hover:underline">Outputs</Link> page.</p>
    </Page>
  );
}
