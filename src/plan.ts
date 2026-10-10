/** Four booleans into an ordered list of passes. The port of `ynonnx_correct.core.plan`. */

import { ConfigurationError } from "@youni/onnxweb-core/core";

import type { CorrectRegistry } from "./registry.js";
import { DESTRUCTIVE, NON_DESTRUCTIVE, NoFeatureSelectedError, enabledFeatures, type CorrectionFeatures, type Feature, type Prefer, type StageKind } from "./types.js";

export interface StagePlan {
  kind: StageKind;
  model: string;
  features: Set<Feature>;
}

const sorted = (features: Iterable<Feature>) => [...features].sort();

/** Raise for features nothing will fix, naming them (`model` null when none was chosen). */
function refuseUnserved(features: ReadonlySet<Feature>, model: string | null, language: string | null = null): void {
  if (features.size === 0) return;
  const reason = model === null ? `no catalogued model serves it for language=${language === null ? "None" : `'${language}'`}` : `${model} does not serve it`;
  throw new ConfigurationError(
    `${sorted(features).join(", ")} was switched on and ${reason}. Nothing downstream can tell a feature that was never applied from one that found nothing to fix, so this is refused before any model is downloaded rather than discovered in the output.`,
  );
}

/** The passes needed to serve `features`, in the order they must run. */
export function buildPlan(
  features: CorrectionFeatures,
  registry: CorrectRegistry,
  options: { punctuationModel?: string; rewriteModel?: string; language?: string | null; prefer?: Prefer } = {},
): StagePlan[] {
  const wanted = enabledFeatures(features);
  if (wanted.size === 0) {
    throw new NoFeatureSelectedError(
      "every feature is switched off, so there is nothing to correct. Turning them all off returns the input unchanged and gives the caller no signal that it did — pass at least one of punctuation, capitalisation, spelling, grammar.",
    );
  }
  const language = options.language ?? null;
  const plans: StagePlan[] = [];

  const tagging = new Set([...wanted].filter((f) => NON_DESTRUCTIVE.has(f)));
  if (tagging.size > 0) {
    let model: string | null;
    if ((options.punctuationModel ?? "auto") === "auto") model = registry.resolvePunctuation(language, options.prefer ?? "balanced");
    else {
      model = options.punctuationModel!;
      registry.warnIfUndeclared(model, language);
    }
    if (model === null) refuseUnserved(tagging, null, language);
    const served = new Set([...registry.featuresOf(model!)].filter((f) => tagging.has(f)));
    if (served.size === 0) refuseUnserved(tagging, model);
    plans.push({ kind: registry.kindOf(model!), model: model!, features: served });
    refuseUnserved(new Set([...tagging].filter((f) => !served.has(f))), model);
  }

  let remaining = new Set([...wanted].filter((f) => DESTRUCTIVE.has(f)));
  let rewriteModel = options.rewriteModel ?? "auto";
  let guard = 0;
  while (remaining.size > 0) {
    guard += 1;
    if (guard > DESTRUCTIVE.size) refuseUnserved(remaining, "any catalogued model");
    let model: string | null;
    if (rewriteModel === "auto") model = registry.resolveRewrite(remaining, language);
    else {
      model = rewriteModel;
      registry.warnIfUndeclared(model, language);
    }
    if (model === null) refuseUnserved(remaining, null, language);
    const served = new Set([...registry.featuresOf(model!)].filter((f) => remaining.has(f)));
    if (served.size === 0) refuseUnserved(remaining, model);
    plans.push({ kind: registry.kindOf(model!), model: model!, features: served });
    remaining = new Set([...remaining].filter((f) => !served.has(f)));

    rewriteModel = "auto";
  }
  return plans;
}

/** One line per pass, for a log or an error. */
export function describePlan(plans: readonly StagePlan[]): string {
  return plans.map((p) => `${p.kind}:${p.model}[${sorted(p.features).join(",")}]`).join(" -> ");
}
