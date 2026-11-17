import { Download } from "lucide-react";
import { useAudit } from "../audit-context";
import { api } from "../api";
import { HealthReport } from "../components/HealthReport";
import { ButtonLink, EmptyState, Page } from "../components/ui";

/** The baseline health report captured when the file was uploaded. It is read from the stored snapshot, never recomputed,
 *  so it keeps describing the raw file after any cleaning. */
export function Baseline() {
  const a = useAudit();
  if (!a.baseline)
    return (
      <Page title="Baseline health report">
        <EmptyState title="No baseline was captured">The baseline health report could not be built for this upload. The audit itself is unaffected.</EmptyState>
      </Page>
    );
  return (
    <Page title="Baseline health report" subtitle="A diagnostic snapshot of the raw file as uploaded, before any cleaning or transformation. Compare it with the audit and the cleaned files.">
      <HealthReport
        snapshot={a.baseline}
        target={a.baseline_target}
        auditTarget={a.overview.target}
        actions={<>
          <ButtonLink href={api.downloadUrl(a.id, "baseline_health_report.csv")} size="sm"><Download className="h-3.5 w-3.5" /> CSV</ButtonLink>
          <ButtonLink href={api.downloadUrl(a.id, "baseline_health_report.json")} size="sm"><Download className="h-3.5 w-3.5" /> JSON</ButtonLink>
        </>}
      />
    </Page>
  );
}
