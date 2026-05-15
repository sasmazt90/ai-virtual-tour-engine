// Main entry point for contract templates
// Re-exports all functionality from the refactored modules

export {
  TEMPLATE_DEFS,
  getTemplateDef,
  getMissingFields,
} from "./contractTemplates/templateDefinitions";

export {
  withPdfSystemState,
  withSignatureDefaults,
  appendAudit,
} from "./contractTemplates/fieldHelpers";

export { buildContractHtml } from "./contractTemplates/htmlTemplates";

export {
  tryGenerateFillablePdfFromContractData,
  tryGeneratePdfFromHtml,
} from "./contractTemplates/pdfGeneration";
