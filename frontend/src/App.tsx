import { useQuery } from "@tanstack/react-query";
import { useEffect } from "react";
import { Link, Outlet, Route, Routes, useParams } from "react-router-dom";
import { api } from "./api";
import { Shell } from "./components/Shell";
import { Button, EmptyState, Skeleton } from "./components/ui";
import { rememberAudit } from "./lib";
import { Baseline } from "./pages/Baseline";
import { Calibration } from "./pages/Calibration";
import { Impact } from "./pages/Impact";
import { Outputs } from "./pages/Outputs";
import { Overview } from "./pages/Overview";
import { Features } from "./pages/Features";
import { Quality } from "./pages/Quality";
import { Review } from "./pages/Review";
import { Readiness } from "./pages/Readiness";
import { RepairPlan } from "./pages/RepairPlan";
import { Research } from "./pages/Research";
import { Start } from "./pages/Start";

function AuditLayout() {
  const { id = "" } = useParams();
  const q = useQuery({ queryKey: ["audit", id], queryFn: () => api.audit(id) });
  useEffect(() => {
    if (q.data) rememberAudit(id);
  }, [q.data, id]);

  if (q.isLoading)
    return (
      <div className="mx-auto max-w-[1200px] space-y-4 p-8">
        <Skeleton className="h-8 w-64" />
        <div className="grid grid-cols-2 gap-4 lg:grid-cols-5">
          {[0, 1, 2, 3, 4].map((i) => (
            <Skeleton key={i} className="h-24" />
          ))}
        </div>
        <Skeleton className="h-72" />
      </div>
    );
  if (q.isError || !q.data)
    return (
      <div className="mx-auto max-w-2xl p-8">
        <EmptyState
          title="This audit is not available"
          action={
            <Link to="/">
              <Button variant="primary">Start a new audit</Button>
            </Link>
          }
        >
          {q.error instanceof Error ? q.error.message : "Unknown error"}. Audits are kept in memory while the API runs, so restarting the API clears them.
        </EmptyState>
      </div>
    );
  return <Outlet context={q.data} />;
}

function NotFound() {
  return (
    <div className="mx-auto max-w-2xl p-8">
      <EmptyState title="Page not found" action={<Link to="/"><Button variant="primary">Go to the start page</Button></Link>} />
    </div>
  );
}

export default function App() {
  return (
    <Routes>
      <Route element={<Shell />}>
        <Route index element={<Start />} />
        <Route path="audit/:id" element={<AuditLayout />}>
          <Route index element={<Overview />} />
          <Route path="baseline" element={<Baseline />} />
          <Route path="quality" element={<Quality />} />
          <Route path="features" element={<Features />} />
          <Route path="calibration" element={<Calibration />} />
          <Route path="readiness" element={<Readiness />} />
          <Route path="impact" element={<Impact />} />
          <Route path="repair" element={<RepairPlan />} />
          <Route path="review" element={<Review />} />
          <Route path="outputs" element={<Outputs />} />
        </Route>
        <Route path="research" element={<Research />} />
        <Route path="*" element={<NotFound />} />
      </Route>
    </Routes>
  );
}
