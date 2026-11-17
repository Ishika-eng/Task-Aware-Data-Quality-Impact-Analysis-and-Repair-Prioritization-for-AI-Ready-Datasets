// Typed client for the TaskClean API (backend/main.py). Every number shown in the UI comes from these responses.

export interface ColumnInfo {
  name: string;
  kind: "numeric" | "text" | "boolean" | "datetime";
  n_unique: number;
  n_missing: number;
  target_candidate: boolean;
  examples: string[];
}

export interface DatasetInfo {
  id: string;
  name: string;
  rows: number;
  n_columns: number;
  input_format: { format?: string; encoding?: string; delimiter?: string; decimal?: string; sheet?: string; sheets?: string[]; rows: number; columns: number };
  columns: ColumnInfo[];
  suggested_target: string | null;
  preview: Record<string, unknown>[];
  demo: boolean;
}

export interface Job {
  id: string;
  status: "queued" | "running" | "done" | "error";
  progress: number;
  message: string;
  stage: string;
  error: string | null;
  audit_id: string | null;
  elapsed: number;
  calibrate: boolean;
}

export type Task = "classification" | "regression";

export interface AuditRequest {
  dataset_id: string;
  target: string | null;
  task: Task;
  include_feature_anomalies: boolean;
  include_joint_outliers: boolean;
  calibrate: boolean;
  model_panel: boolean;
  seeds: number;
  max_model_rows: number;
}

export interface PanelModel { key: string; label: string; role: string; noise_floor: number | null; noise_floor_auc: number | null }
export interface PanelArm { key: string; label: string; text: string; rows_removed: number; cells_changed: number }
export interface PanelCell {
  rmse_mean?: number; delta_rmse?: number;
  model: string; arm: string; n_seeds: number; f1_mean: number; f1_std: number; auc_mean: number | null; acc_mean: number;
  f1_noleak_mean: number | null; learnable: boolean; noise_floor: number; delta_f1: number; delta_auc: number | null;
  delta_pct: number | null; seeds_agree: number; verdict: "baseline" | "improved" | "worse" | "within noise" | "no change" | "no effect" | "not enough runs"; auc_verdict?: string;
}
export interface ProofModel { model: string; label: string; noise_floor: number; clean_f1: number; dirty_f1: number; damage: number; damage_beyond_noise: boolean }
export interface ProofCell {
  model: string; arm: string; f1_mean: number; f1_std: number; auc_mean: number; n_seeds: number; delta_f1: number | null;
  verdict: "ceiling" | "baseline" | "improved" | "worse" | "within noise" | "no change" | "no effect"; share_recovered: number | null; seeds_agree: number | null;
}
export interface Proof {
  status: "none" | "queued" | "running" | "ok" | "error"; progress?: number; message?: string; reason?: string; source?: string;
  seeds?: number[]; n_seeds?: number; rows?: number; rate_per_error_type?: number; metric?: string; seconds?: number; select_margin?: number;
  models?: ProofModel[]; arms?: { key: string; label: string; text: string }[]; results?: ProofCell[];
  criteria?: Record<string, { pass: boolean } & Record<string, unknown>>; tuned_choices?: Record<string, Record<string, number>>;
  default_issues?: string[]; caveat?: string; task?: Task; dataset?: string;
}

export interface ModelPanel {
  status: "none" | "queued" | "running" | "ok" | "declined" | "error";
  task?: Task;
  progress?: number; message?: string; reason?: string; seconds?: number;
  metric?: string; seeds?: number[]; n_seeds?: number; rows_used?: number; train_rows?: number; test_rows?: number; n_classes?: number; chance_f1?: number;
  models?: PanelModel[]; arms?: PanelArm[]; results?: PanelCell[];
  interaction?: { model: string; aggressive: number; sum_of_singles: number }[];
  leakage?: { test_rows_with_exact_copy_in_train_pct: number; note: string };
  default_issues?: string[]; notes?: string[]; caveat?: string; settings?: Record<string, number>;
}

export interface Overview {
  rows_uploaded: number;
  rows_analysed: number;
  rows_without_target_excluded_from_label_checks: number;
  columns: number;
  target: string | null;
  task: string;
  n_classes: number | null;
  notes: string[];
  model_based_detectors_ran_on_rows: number;
  model_based_detectors_sampled: boolean;
  columns_not_analysed: Record<string, string>;
  columns_skipped_by_model_based_detectors: Record<string, string>;
  model_used_for_benchmark_validation: string;
  audit_seconds: number;
  detector_seconds: Record<string, number>;
  generated_at: string;
}

export interface QualityRow {
  dimension: string;
  issue_key: string;
  detector_type: string;
  detected_rate: number | null;
  detected_count: number | null;
  count_unit: string;
  benchmark_precision: number | null;
  benchmark_recall: number | null;
  reliability_source: string | null;
  evidence_status: string;
  note: string;
}

export interface PolicyRow {
  issue_key: string;
  issue: string;
  detector_ran: boolean;
  detected_rate: number | null;
  n_repair_candidates: number;
  n_flag_only: number;
  repairability_status: string;
  benchmark_risk: number | null;
  benchmark_repair_precision: number | null;
  safe_auto: boolean;
  default_action: string;
  evidence_source: string;
  /** Phase 16: "agree" | "model-dependent" | "not tested" (Random Forest vs gradient boosting, Adult-sourced evidence only). */
  model_check: string;
  model_check_detail: string;
  reason: string;
}

export interface ImpactRow {
  issue: string;
  issue_key: string;
  detected_rate: number | null;
  benchmark_impact_estimate: number | null;
  is_boundary_estimate: boolean;
  impact_source: string;
  impact_unit: string;
  impact_estimate_auc: number | null;
  metrics_disagree: boolean;
  metric_note: string;
  impact_model_note: string;
  repairability_status: string;
  benchmark_risk: number | null;
  policy_decision: string;
  recommendation: string;
}

export interface ReadinessRow {
  dimension: string;
  issue_key: string;
  observed_rate: number | null;
  baseline_false_positive_floor: number | null;
  baseline_source: string | null;
  evidence_status: string;
  evidence_note: string;
  status: string;
  rate_after_cleaning: number | null;
}

export interface Summary {
  rows_uploaded: number;
  rows_after_cleaning: number;
  rows_dropped: number;
  cells_modified: number;
  repairs_applied_by_issue: Record<string, number>;
  flagged_for_review: number;
  flagged_only: number;
  skipped_conflicts: number;
  skipped_row_dropped: number;
  issues_applied: string[];
  human_overrides: string[];
}

export interface EvidenceRow {
  error_type: string;
  rate: number;
  mean_damage: number;
  mean_damage_auc: number | null;
  damage_std: number | null;
  repair_precision: number | null;
  repair_recall: number | null;
  mean_recovery: number;
  repairability_status: string;
  status_macrof1: string;
  status_auc: string;
  material_risk: number;
  underpowered: boolean;
  restored_exactly: boolean;
}

export interface CurveRow {
  error_type: string;
  rate: number;
  mean_damage: number;
  damage_std: number | null;
  mean_damage_auc?: number | null;
}

export interface Calibration {
  task?: Task;
  status: "not_run" | "declined" | "ok";
  reason?: string;
  metric?: string;
  n_seeds?: number;
  rates?: number[];
  seconds?: number;
  reference?: {
    rows_labeled: number;
    rows_complete: number;
    rows_after_dedup: number;
    rows_used: number;
    rows_train: number;
    rows_test: number;
    rows_removed_label_problems: number;
    columns_left_out: Record<string, string>;
    columns_used: string[];
    minority_class_share?: number;
  };
  noise_floor?: number;
  noise_floor_auc?: number | null;
  has_auc?: boolean;
  clean_macro_f1?: number;
  chance_f1?: number;
  minority_share?: number | null;
  imbalance_warning?: boolean;
  underpowered?: boolean;
  evidence?: EvidenceRow[];
  impact_curve?: CurveRow[];
  detector_reliability?: { error_type: string; unit: string; precision: number; recall: number }[];
  detector_tracking?: { error_type: string; evidence_status: string; note: string; pearson_r: number | null }[];
  baseline_floors?: Record<string, number>;
  skipped_issues?: Record<string, string>;
  metric_disagreement?: { error_type: string; disagree: boolean; note: string }[];
}

export interface HealthColumn {
  name: string;
  kind: string;
  n_unique: number;
  missing: number;
  blank_cells: number;
  placeholder_cells: number;
  missing_pct: number;
  examples: string[];
  flags: string[];
  min?: number | null;
  max?: number | null;
  mean?: number | null;
  median?: number | null;
  std?: number | null;
  skew?: number | null;
  top_values?: { value: string; share: number }[];
}

export interface OutlierBlock {
  outlier_rows: number;
  outlier_pct: number;
  per_column: Record<string, number>;
  rule: string;
  percentile_rule_columns?: string[];
}

/** The baseline health report: a snapshot of the raw upload, captured before any cleaning (docs/PHASE18_SPEC.md). */
export interface HealthSnapshot {
  version: number;
  captured_at_utc: string;
  overview: {
    file_name: string;
    file_bytes: number | null;
    file_mb: number | null;
    sha256: string | null;
    rows: number;
    columns: number;
    memory_mb: number;
    input_format: Record<string, unknown>;
    type_counts: Record<string, number>;
  };
  columns: HealthColumn[];
  completeness: {
    total_cells: number;
    missing_cells: number;
    blank_cells: number;
    placeholder_cells: number;
    missing_pct: number;
    rows_with_missing: number;
    rows_with_missing_pct: number;
    columns_entirely_missing: string[];
    placeholder_tokens: string[];
  };
  duplicates: {
    exact_rows: number;
    exact_pct: number;
    exact_examples: number[];
    near_extra_rows: number;
    near_extra_pct: number;
    near_examples: number[];
    near_definition: string;
  } | null;
  anomalies: {
    iqr: OutlierBlock | null;
    zscore: OutlierBlock | null;
    skewed_columns: { column: string; skew: number }[];
    constant_columns: string[];
    possible_identifier_columns: string[];
  };
  inconsistencies: {
    rows: number;
    rows_pct: number;
    cells: number;
    by_column: {
      column: string;
      cells: number;
      by_rule: Record<string, number>;
      examples: { value: string; canonical: string | null; rule: string }[];
    }[];
    by_rule: Record<string, number>;
    rule_labels: Record<string, string>;
    skipped_columns: Record<string, string>;
  } | null;
  notes: string[];
  not_covered: string[];
}

export interface TargetProfile {
  target: string;
  rows: number;
  rows_without_target: number;
  rows_without_target_pct: number;
  n_classes: number;
  classes: { value: string; rows: number; share: number }[];
  usable: boolean;
  warnings: string[];
  imbalance_ratio?: number;
  minority_share?: number;
  rare_classes?: string[];
  /** regression targets (Phase 21) */
  task?: Task; n_distinct?: number; non_numeric_values?: number;
  mean?: number; std?: number; min?: number; q25?: number; median?: number; q75?: number; max?: number; skew?: number;
}

export interface Audit {
  id: string;
  dataset_id: string;
  dataset_name: string;
  request: AuditRequest;
  version: string;
  overview: Overview;
  quality: QualityRow[];
  policy: PolicyRow[];
  impact: ImpactRow[];
  impact_note: string;
  readiness: ReadinessRow[];
  limitations: string[];
  calibration: Calibration;
  selection: { applied: string[]; default: string[]; label: string };
  summary: { evidence_based: Summary; aggressive: Summary };
  review: ReviewSummary;
  features: FeatureRow[] | null;
  files: { name: string; bytes: number }[];
  never_auto_apply: string[];
  /** Phase 23 A1: proposals left in the review queue because the calibration did not cover their column, per issue type. */
  held_back?: Record<string, { count: number; columns: Record<string, number> }>;
  baseline: HealthSnapshot | null;
  baseline_target: TargetProfile | null;
}

export interface FeatureRow {
  column: string;
  kind: "numeric" | "categorical";
  verdict: "relevant" | "constant" | "id_like" | "redundant" | "no_detectable_relevance" | "not_assessed";
  reason: string;
  redundant_with: string | null;
  perm_importance: number | null;
  perm_importance_std: number | null;
  mutual_info: number | null;
  probe_importance_threshold: number | null;
  probe_mi_threshold: number | null;
  recommendation: string;
}

export interface ReviewSummary {
  proposed: number;
  flag_only: number;
  total: number;
  by_issue: { issue_key: string; problem: string; proposed: number; flag_only: number }[];
}

export interface RepairLogRow {
  repair_id: string;
  row_id: number;
  column: string | null;
  problem: string;
  original_value: unknown;
  proposed_value: unknown;
  applied_value: unknown;
  action: string;
  method: string;
  tier: string;
  detector_score: number | null;
  repair_evidence_status: string;
  human_override: boolean;
}

export interface RepairLogPage {
  total: number;
  page: number;
  page_size: number;
  rows: RepairLogRow[];
  actions: Record<string, number>;
  problems: Record<string, number>;
}

export type Rec = Record<string, any>;
export interface Benchmark {
  baseline: Rec[];
  impact: Rec[];
  detection: Rec[];
  repair: Rec[];
  evidence: Rec[];
  priority: Rec[];
  sensitivity: Rec[];
  tracking: Rec[];
  correlation: Rec[];
  baseline_rates: Rec;
  model_check: { evidence: Rec[]; priority: Rec[]; damage: Rec[]; criteria: Rec };
  bank: { posf1: Rec[]; macrof1: Rec[]; auc: Rec[]; decisions: Rec[]; damage_corr: Rec[]; regimes: Rec[] };
}

export interface AppConfig {
  issues: string[];
  problem_text: Record<string, string>;
  never_auto_apply: string[];
  safe_repair_min_precision: number;
  max_certifiable_noise_floor: number;
  max_model_rows: number;
  max_classes: number;
  max_cat_levels: number;
  null_drop_fraction: number;
  imbalance_warning_share: number;
}

async function parse<T>(r: Response): Promise<T> {
  if (!r.ok) {
    let detail = `${r.status} ${r.statusText}`;
    try {
      const body = await r.json();
      if (body?.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* not JSON */
    }
    throw new Error(detail);
  }
  return r.json() as Promise<T>;
}

const json = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "content-type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  health: () => fetch("/api/health").then((r) => parse<{ ok: boolean; version: string; demo_available: boolean }>(r)),
  config: () => fetch("/api/config").then((r) => parse<AppConfig>(r)),
  uploadDataset: (file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    return fetch("/api/datasets/upload", { method: "POST", body: fd }).then((r) => parse<DatasetInfo>(r));
  },
  datasetHealth: (id: string, target: string | null, task: Task = "classification") =>
    fetch(`/api/datasets/${id}/health${target ? `?target=${encodeURIComponent(target)}&task=${task}` : ""}`).then((r) =>
      parse<{ snapshot: HealthSnapshot; target: TargetProfile | null; suggested_task?: { task: Task; reason: string } }>(r)),
  pickSheet: (id: string, sheet: string) => fetch(`/api/datasets/${id}/sheet`, json({ sheet })).then((r) => parse<DatasetInfo>(r)),
  demoDataset: () => fetch("/api/datasets/demo", { method: "POST" }).then((r) => parse<DatasetInfo>(r)),
  startAudit: (req: AuditRequest) => fetch("/api/audits", json(req)).then((r) => parse<{ job_id: string }>(r)),
  job: (id: string) => fetch(`/api/jobs/${id}`).then((r) => parse<Job>(r)),
  audit: (id: string) => fetch(`/api/audits/${id}`).then((r) => parse<Audit>(r)),
  apply: (id: string, issues: string[]) =>
    fetch(`/api/audits/${id}/apply`, json({ issues })).then((r) => parse<Audit>(r)),
  repairLog: (id: string, p: { variant: string; action?: string; issue?: string; page: number; page_size: number }) => {
    const q = new URLSearchParams({ variant: p.variant, page: String(p.page), page_size: String(p.page_size) });
    if (p.action) q.set("action", p.action);
    if (p.issue) q.set("issue", p.issue);
    return fetch(`/api/audits/${id}/repair-log?${q}`).then((r) => parse<RepairLogPage>(r));
  },
  benchmark: () => fetch("/api/benchmark").then((r) => parse<Benchmark>(r)),
  downloadUrl: (id: string, name: string) => `/api/audits/${id}/download/${name}`,
  reviewCsvUrl: (id: string, variant: string) => `/api/audits/${id}/review.csv?variant=${variant}`,
  zipUrl: (id: string) => `/api/audits/${id}/download.zip`,
  proof: (task: Task = "classification") => fetch(`/api/proof?task=${task}`).then((r) => parse<Proof>(r)),
  startProof: (seeds: number, task: Task = "classification") => fetch("/api/proof", json({ seeds, task })).then((r) => parse<{ status: string }>(r)),
  panel: (id: string) => fetch(`/api/audits/${id}/panel`).then((r) => parse<ModelPanel>(r)).then(normalisePanel),
  startPanel: (id: string, seeds: number) => fetch(`/api/audits/${id}/panel`, json({ seeds })).then((r) => parse<{ status: string }>(r)),
};

/** A regression panel (Phase 21) names its score fields score_*; map them onto the fields the page already reads, so one page
 * serves both tasks. The page switches its labels on `task`. */
function normalisePanel(p: ModelPanel): ModelPanel {
  if (p.task !== "regression" || !p.results) return p;
  type Raw = PanelCell & { score_mean?: number; score_std?: number; delta_score?: number; score_noleak_mean?: number | null };
  return { ...p, results: (p.results as Raw[]).map((c) => ({
    ...c, f1_mean: c.score_mean ?? 0, f1_std: c.score_std ?? 0, delta_f1: c.delta_score ?? 0, f1_noleak_mean: c.score_noleak_mean ?? null,
    auc_mean: c.rmse_mean ?? null, delta_auc: c.delta_rmse ?? null, acc_mean: 0, delta_pct: null,
  })) };
}
