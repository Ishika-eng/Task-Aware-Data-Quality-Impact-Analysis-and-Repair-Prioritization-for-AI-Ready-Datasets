import { useQuery } from "@tanstack/react-query";
import { Bar, BarChart, CartesianGrid, Cell, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, type Rec } from "../api";
import { axisLine, axisTick, ChartTip, gridProps } from "../components/charts";
import { Alert, Card, DataTable, Dot, Page, Skeleton, Stat, Status, type Col } from "../components/ui";
import { ISSUE_COLOR, ISSUE_NAME, ISSUE_ORDER, num, pct, signed } from "../lib";

const RATES = [0.05, 0.1, 0.2];
const rateName = (r: number) => `${Math.round(r * 100)}%`;
const find = (rows: Rec[], k: string, rate?: number) => rows.find((r) => r.error_type === k && (rate === undefined || r.rate === rate));

export function Research() {
  const q = useQuery({ queryKey: ["benchmark"], queryFn: api.benchmark, staleTime: Infinity });
  if (q.isLoading) return <div className="mx-auto max-w-[1200px] space-y-4 p-8"><Skeleton className="h-8 w-72" /><Skeleton className="h-72" /><Skeleton className="h-72" /></div>;
  if (!q.data) return <Page title="Research results"><Alert tone="bad" title="Could not load the research results">{(q.error as Error)?.message}</Alert></Page>;
  const b = q.data;
  const rf = b.baseline.find((r) => r.model === "random_forest");
  const lr = b.baseline.find((r) => r.model === "logistic_regression");

  const damage = RATES.map((rate) => ({ rate: rateName(rate), ...Object.fromEntries(ISSUE_ORDER.map((k) => [k, find(b.impact, k, rate)?.mean_damage])) }));
  const damage20 = ISSUE_ORDER.map((k) => ({ key: k, name: ISSUE_NAME[k], damage: find(b.impact, k, 0.2)?.mean_damage as number }));
  const detection = ["missing_values", "duplicates", "label_errors", "outliers", "feature_corruption"].map((k) => {
    const d = b.detection.find((r) => r.error_type === k)!;
    return { name: `${ISSUE_NAME[k]} (${d.unit})`, Precision: d.precision, Recall: d.recall };
  });
  const recovery = ISSUE_ORDER.map((k) => ({ name: ISSUE_NAME[k], ...Object.fromEntries(RATES.map((r) => [rateName(r), (find(b.repair, k, r)?.mean_recovery ?? 0) * 100])) }));
  const repair10 = ISSUE_ORDER.map((k) => { const r = find(b.repair, k, 0.1); return { name: ISSUE_NAME[k], Precision: r?.mean_repair_precision, Recall: r?.mean_repair_recall }; });
  const priority = [...b.priority].sort((x, y) => y.priority_score - x.priority_score).map((r) => ({ key: r.error_type, name: ISSUE_NAME[r.error_type], score: r.priority_score }));
  const mc = b.model_check;
  const rank10 = (model: string) => mc.priority.filter((r) => r.model === model && r.rate === 0.1).sort((x, y) => x.rank - y.rank).map((r) => ISSUE_NAME[r.error_type]);
  const CRITERIA: [string, string][] = [
    ["G0", "Harness reproduces the frozen Random Forest numbers"], ["G1", "Gradient boosting is a competent model (clean F1)"],
    ["G2", "Damage ordering transfers (label errors largest)"], ["G3", "Damage size transfers (median ratio 0.5–2.0)"],
    ["G4", "Repair status agrees (11 of 15 cells)"], ["G5", "Duplicate removal passes the safe-auto gate at every rate"],
    ["G6", "No gradient-boosting-only harm"], ["G7", "Priority ranking transfers"],
    ["G8", "Label errors dominate at every model capacity"], ["G9", "Bank Marketing agrees (macro-F1)"],
  ];
  const passed = CRITERIA.filter(([k]) => mc.criteria[k]?.pass === true).length;
  const bank = ISSUE_ORDER.map((k) => ({
    name: ISSUE_NAME[k],
    "Positive-class F1": find(b.bank.posf1, k, 0.2)?.mean_damage, "Macro-F1": find(b.bank.macrof1, k, 0.2)?.mean_damage, "ROC-AUC": find(b.bank.auc, k, 0.2)?.mean_damage,
  }));

  const evCols: Col<Rec>[] = [
    { key: "e", header: "Error", cell: (r) => <span className="flex items-center gap-2 font-medium"><Dot color={ISSUE_COLOR[r.error_type]} />{ISSUE_NAME[r.error_type]}</span> },
    { key: "r", header: "Rate", align: "right", cell: (r) => rateName(r.rate) },
    { key: "d", header: "Damage", align: "right", cell: (r) => num(r.mean_damage) },
    { key: "p", header: "Repair precision", align: "right", cell: (r) => pct(r.repair_precision, 0) },
    { key: "rec", header: "Recovery", align: "right", cell: (r) => signed(r.mean_recovery) },
    { key: "seeds", header: "Seeds + / −", align: "right", cell: (r) => `${r.n_pos_seeds} / ${r.n_neg_seeds}` },
    { key: "s", header: "Status", cell: (r) => <Status value={r.repairability_status} /> },
    { key: "risk", header: "Risk", align: "right", cell: (r) => num(r.risk) },
  ];
  const decCols: Col<Rec>[] = [
    { key: "s", header: "Evidence source", cell: (r) => r.source },
    { key: "a", header: "Compared with", cell: (r) => <span className="text-muted">{r.against}</span> },
    { key: "g", header: "Agree", align: "right", cell: (r) => r.agree },
    { key: "f", header: "False approvals", align: "right", cell: (r) => r.false_approvals },
    { key: "m", header: "Missed safe repairs", align: "right", cell: (r) => r.missed_safe_repairs },
  ];
  const evSorted = [...b.evidence].sort((x, y) => ISSUE_ORDER.indexOf(x.error_type) - ISSUE_ORDER.indexOf(y.error_type) || x.rate - y.rate);

  return (
    <Page title="Research results" subtitle="The frozen UCI Adult measurements the policy falls back on when you do not calibrate, plus the Bank Marketing validation. Random Forest, 300 trees, errors injected into training data only, seeds 42 to 46.">
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <Stat label="Clean F1 (RF)" value={num(rf?.f1, 3)} hint="positive class, test set" />
        <Stat label="Accuracy" value={num(rf?.accuracy, 3)} />
        <Stat label="ROC-AUC" value={num(rf?.roc_auc, 3)} />
        <Stat label="Logistic Regression F1" value={num(lr?.f1, 3)} hint="baseline only" />
      </div>

      <Card title="Impact: how much does each error hurt?" description="Mean F1 damage against the clean baseline, paired by seed. The dashed line is the 0.003 noise floor.">
        <div className="grid gap-6 lg:grid-cols-[minmax(0,1.5fr)_minmax(0,1fr)]">
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={damage} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
                <CartesianGrid {...gridProps} />
                <XAxis dataKey="rate" tick={axisTick} axisLine={axisLine} tickLine={false} />
                <YAxis tick={axisTick} axisLine={false} tickLine={false} tickFormatter={(v) => v.toFixed(2)} />
                <Tooltip content={<ChartTip fmt={(v) => num(v)} />} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <ReferenceLine y={0.003} stroke="var(--muted)" strokeDasharray="4 4" />
                {ISSUE_ORDER.map((k) => <Line isAnimationActive={false} key={k} type="monotone" dataKey={k} name={ISSUE_NAME[k]} stroke={ISSUE_COLOR[k]} strokeWidth={2} dot={{ r: 3 }} />)}
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={damage20} layout="vertical" margin={{ top: 8, right: 16, left: 24, bottom: 0 }}>
                <CartesianGrid {...gridProps} horizontal={false} vertical />
                <XAxis type="number" tick={axisTick} axisLine={axisLine} tickLine={false} />
                <YAxis type="category" dataKey="name" tick={axisTick} axisLine={false} tickLine={false} width={110} />
                <Tooltip cursor={{ fill: "var(--surface-2)" }} content={<ChartTip fmt={(v) => num(v)} />} />
                <Bar isAnimationActive={false} dataKey="damage" name="Damage at 20%" radius={[0, 4, 4, 0]}>{damage20.map((d) => <Cell key={d.key} fill={ISSUE_COLOR[d.key]} />)}</Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
        <p className="mt-3 text-sm text-muted">Label errors dominate (0.0098 → 0.0649 over 5–20%). Duplicates, missing values and outliers stay near the noise floor for a Random Forest. These are Random Forest numbers: other models can react very differently.</p>
      </Card>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card title="Detection: can each error be found without ground truth?" description="Precision and recall of the blind detectors at 10% corruption, graded against the injected-error log.">
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={detection} margin={{ top: 8, right: 8, left: -8, bottom: 28 }}>
                <CartesianGrid {...gridProps} />
                <XAxis dataKey="name" tick={axisTick} axisLine={axisLine} tickLine={false} interval={0} angle={-18} textAnchor="end" height={56} />
                <YAxis domain={[0, 1]} tick={axisTick} axisLine={false} tickLine={false} tickFormatter={(v) => `${Math.round(v * 100)}%`} />
                <Tooltip cursor={{ fill: "var(--surface-2)" }} content={<ChartTip fmt={(v) => pct(v, 0)} />} />
                <Legend verticalAlign="top" wrapperStyle={{ fontSize: 12 }} />
                <Bar isAnimationActive={false} dataKey="Precision" fill="var(--c-series-a)" radius={[4, 4, 0, 0]} />
                <Bar isAnimationActive={false} dataKey="Recall" fill="var(--c-series-b)" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
        <Card title="Repair: does automatic repair help?" description="F1 recovery (points ×100) after real repair. Anything within ±0.3 is noise; feature repair is net-harmful at every rate.">
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={recovery} margin={{ top: 8, right: 8, left: -8, bottom: 28 }}>
                <CartesianGrid {...gridProps} />
                <XAxis dataKey="name" tick={axisTick} axisLine={axisLine} tickLine={false} interval={0} angle={-18} textAnchor="end" height={56} />
                <YAxis tick={axisTick} axisLine={false} tickLine={false} />
                <Tooltip cursor={{ fill: "var(--surface-2)" }} content={<ChartTip fmt={(v) => v.toFixed(2)} />} />
                <Legend verticalAlign="top" wrapperStyle={{ fontSize: 12 }} />
                <ReferenceLine y={0} stroke="var(--fg)" strokeOpacity={0.5} />
                <Bar isAnimationActive={false} dataKey="5%" fill="var(--c-series-a)" fillOpacity={0.4} radius={[3, 3, 0, 0]} />
                <Bar isAnimationActive={false} dataKey="10%" fill="var(--c-series-a)" fillOpacity={0.7} radius={[3, 3, 0, 0]} />
                <Bar isAnimationActive={false} dataKey="20%" fill="var(--c-series-a)" radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card title="Repair correctness at 10%" description="Repair precision and recall. The most damaging error (labels) is the hardest to repair automatically: precise, but it repairs very few.">
          <div className="h-60">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={repair10} margin={{ top: 8, right: 8, left: -8, bottom: 0 }}>
                <CartesianGrid {...gridProps} />
                <XAxis dataKey="name" tick={axisTick} axisLine={axisLine} tickLine={false} interval={0} />
                <YAxis domain={[0, 1]} tick={axisTick} axisLine={false} tickLine={false} tickFormatter={(v) => `${Math.round(v * 100)}%`} />
                <Tooltip cursor={{ fill: "var(--surface-2)" }} content={<ChartTip fmt={(v) => pct(v, 0)} />} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Bar isAnimationActive={false} dataKey="Precision" fill="var(--c-series-a)" radius={[4, 4, 0, 0]} />
                <Bar isAnimationActive={false} dataKey="Recall" fill="var(--c-series-b)" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
        <Card title="Repair priority at 10%" description="Random Forest. Priority = impact − risk + repairability, each rescaled within the rate. Label errors rank first and feature corruption last in every one of 15 weight combinations at 10%. Under gradient boosting the top three are the same but missing values ranks last: a model-dependent ranking (see the model check below).">
          <div className="h-60">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={priority} layout="vertical" margin={{ top: 8, right: 16, left: 24, bottom: 0 }}>
                <CartesianGrid {...gridProps} horizontal={false} vertical />
                <XAxis type="number" tick={axisTick} axisLine={axisLine} tickLine={false} />
                <YAxis type="category" dataKey="name" tick={axisTick} axisLine={false} tickLine={false} width={110} />
                <Tooltip cursor={{ fill: "var(--surface-2)" }} content={<ChartTip fmt={(v) => signed(v, 2)} />} />
                <ReferenceLine x={0} stroke="var(--fg)" strokeOpacity={0.5} />
                <Bar isAnimationActive={false} dataKey="score" name="Priority" radius={[0, 4, 4, 0]}>{priority.map((p) => <Cell key={p.key} fill={ISSUE_COLOR[p.key]} />)}</Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>

      <Card title="Model check: is this about the errors or about the model? (Phase 16)" description={`The key results were re-measured with gradient boosting (only the downstream model changed). ${passed} of ${CRITERIA.length} pre-registered criteria passed.`}>
        <div className="grid gap-4 lg:grid-cols-2">
          <ul className="space-y-1.5 text-sm">
            {CRITERIA.map(([k, text]) => {
              const ok = mc.criteria[k]?.pass === true;
              return <li key={k} className="flex items-start gap-2"><span className={`mt-0.5 inline-flex h-5 w-9 shrink-0 items-center justify-center rounded text-[11px] font-semibold ${ok ? "bg-good-soft text-good" : "bg-bad-soft text-bad"}`}>{k}</span><span className="flex-1">{text}</span><span className={`shrink-0 text-xs font-medium ${ok ? "text-good" : "text-bad"}`}>{ok ? "pass" : "fail"}</span></li>;
            })}
          </ul>
          <div className="space-y-3 text-[13px] leading-relaxed text-muted">
            <p><b className="text-fg">Held up:</b> label errors are the largest damage under both models and every capacity tried; feature-corruption repair is net-harmful under both; the top three priorities are the same.</p>
            <p><b className="text-fg">Did not hold up:</b> the size of the damage (about 0.35× the forest's), several repair verdicts, duplicate removal's automatic approval, and the bottom of the priority ranking. On Bank Marketing the macro-F1 comparison did not replicate.</p>
            <div className="rounded-lg border border-line p-3">
              <div className="text-xs uppercase tracking-wide">Priority at 10%, equal weights</div>
              <div className="mt-1 text-fg"><b>Random Forest:</b> {rank10("RF").join(" > ")}</div>
              <div className="text-fg"><b>Gradient boosting:</b> {rank10("GB").join(" > ")}</div>
              <div className="mt-1">Model-dependent ranking: treat the order of the bottom two as unsettled.</div>
            </div>
          </div>
        </div>
      </Card>

      <Card title="Does a detector's flagged rate track real damage?" description="Detected rate (x) against measured damage (y) at 5%, 10% and 20% true corruption. Only three points per error type: directional, not statistically significant.">
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-5">
          {ISSUE_ORDER.map((k) => {
            const pts = b.tracking.filter((t) => t.error_type === k).sort((x, y) => x.true_rate - y.true_rate).map((t) => ({ det: t.detected_rate * 100, dmg: t.measured_damage * 100, rate: rateName(t.true_rate) }));
            const r = b.correlation.find((c) => c.error_type === k)?.pearson_r as number;
            return (
              <div key={k} className="rounded-lg border border-line p-3">
                <div className="flex items-center justify-between text-sm font-medium"><span className="flex items-center gap-2"><Dot color={ISSUE_COLOR[k]} />{ISSUE_NAME[k]}</span><span className={r < 0 ? "text-bad" : "text-muted"}>r = {r.toFixed(2)}</span></div>
                <div className="mt-2 h-28">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={pts} margin={{ top: 6, right: 8, left: -22, bottom: 0 }}>
                      <CartesianGrid {...gridProps} />
                      <XAxis dataKey="det" type="number" domain={["dataMin", "dataMax"]} tick={axisTick} axisLine={axisLine} tickLine={false} tickFormatter={(v) => `${Math.round(v)}%`} />
                      <YAxis tick={axisTick} axisLine={false} tickLine={false} />
                      <Tooltip content={<ChartTip fmt={(v) => v.toFixed(2)} title={(_l, p) => `true rate ${p[0]?.payload?.rate}`} />} />
                      <Line isAnimationActive={false} dataKey="dmg" name="damage (F1×100)" stroke={ISSUE_COLOR[k]} strokeWidth={2} dot={{ r: 3 }} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            );
          })}
        </div>
        <Alert tone="warn" title="Label errors: the flagged rate moves opposite to real damage">
          The label detector's rate falls (3.5%, 3.6%, 2.1%) as true corruption rises 5% → 20% and damage grows from 0.0098 to 0.0649. A low flagged label-error rate must never be read as reassurance. Pooled over all 15 points the correlation is r = −0.24 (p = 0.39), which is why no single readiness score is reported.
        </Alert>
      </Card>

      <Card title="Evidence table the policy reads" description="Status is positive or negative only if at least 80% of the 5 seeds agree on the sign and the mean recovery clears the 0.003 noise floor; otherwise weak." pad={false}>
        <div className="p-2"><DataTable columns={evCols} rows={evSorted} rowKey={(r) => `${r.error_type}${r.rate}`} dense /></div>
      </Card>

      <Card title="Second dataset: Bank Marketing (11.7% positive)" description="Damage at 20% corruption under three metrics. On an imbalanced target label noise looks harmless or even helpful under F1, and is the most damaging error under ROC-AUC.">
        <div className="h-72">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={bank} margin={{ top: 8, right: 8, left: -8, bottom: 0 }}>
              <CartesianGrid {...gridProps} />
              <XAxis dataKey="name" tick={axisTick} axisLine={axisLine} tickLine={false} interval={0} />
              <YAxis tick={axisTick} axisLine={false} tickLine={false} />
              <Tooltip cursor={{ fill: "var(--surface-2)" }} content={<ChartTip fmt={(v) => signed(v)} />} />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              <ReferenceLine y={0} stroke="var(--fg)" strokeOpacity={0.5} />
              <Bar isAnimationActive={false} dataKey="Positive-class F1" fill="var(--c-series-b)" radius={[3, 3, 0, 0]} />
              <Bar isAnimationActive={false} dataKey="Macro-F1" fill="var(--c-series-c)" radius={[3, 3, 0, 0]} />
              <Bar isAnimationActive={false} dataKey="ROC-AUC" fill="var(--c-series-a)" radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
        <h3 className="mb-1 mt-5 text-sm font-semibold">Safe-to-auto-repair decisions vs ground truth</h3>
        <p className="mb-2 text-xs text-muted">Nine decidable cells. After the two fixes (ROC-AUC alongside macro-F1, and a power guard) every source ties, so calibration is not shown to beat reusing the Adult evidence for this decision.</p>
        <DataTable columns={decCols} rows={b.bank.decisions} rowKey={(r) => `${r.source}${r.against}`} dense />
      </Card>
    </Page>
  );
}
