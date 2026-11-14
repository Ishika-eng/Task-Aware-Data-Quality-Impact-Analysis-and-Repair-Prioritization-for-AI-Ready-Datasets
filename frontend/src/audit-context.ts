import { useOutletContext } from "react-router-dom";
import type { Audit } from "./api";

/** The audit loaded by AuditLayout, available to every audit page. */
export const useAudit = () => useOutletContext<Audit>();
