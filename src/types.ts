/**
 * Features, configurations, results and errors. The port of `ynonnx_correct.types` and
 * `ynonnx_correct.exceptions`.
 */

import { ConfigurationError, YnONNXError, registerErrors } from "@youni/onnxweb-core/core";

export const FEATURES = ["punctuation", "capitalisation", "spelling", "grammar"] as const;
export type Feature = (typeof FEATURES)[number];

/** The features that may not change a single letter (the verbatim guard reads this). */
export const NON_DESTRUCTIVE: ReadonlySet<Feature> = new Set(["punctuation", "capitalisation"]);
/** The features that rewrite words, and need a model that generates. */
export const DESTRUCTIVE: ReadonlySet<Feature> = new Set(["spelling", "grammar"]);

/** How a model produces its correction: one label per token, or a fresh sequence. */
export type StageKind = "tagging" | "rewrite";
/** Which file a model's vocabulary arrives in. */
export type TokenizerKind = "tokenizers" | "sentencepiece";
/** What happens when a rewrite comes back the wrong length. */
export type LengthGuardAction = "keep_original" | "accept" | "error";
/** Steers `auto`. */
export type Prefer = "fast" | "balanced" | "quality";

export class YnONNXCorrectError extends YnONNXError {
  static override readonly code: string = "YnONNXCorrectError";
}
/** The graph's shape contradicts the stage the catalogue declared for it. */
export class StageMismatchError extends YnONNXCorrectError {
  static override readonly code: string = "StageMismatchError";
}
/** A non-destructive stage changed a word. Raised, never warned. */
export class VerbatimViolationError extends YnONNXCorrectError {
  static override readonly code: string = "VerbatimViolationError";
}
/** A rewrite fell outside the length guard and the policy is `error`. */
export class RewriteLengthError extends YnONNXCorrectError {
  static override readonly code: string = "RewriteLengthError";
}
/** Every switch is off, so there is nothing to correct. */
export class NoFeatureSelectedError extends YnONNXCorrectError {
  static override readonly code: string = "NoFeatureSelectedError";
}
registerErrors(YnONNXCorrectError, StageMismatchError, VerbatimViolationError, RewriteLengthError, NoFeatureSelectedError);

export type CorrectionFeatures = Record<Feature, boolean>;

/** `"FR"`, `"fr-FR"`, `"fr_CA"` → `"fr"`; null and `"auto"` unchanged. */
export function normaliseLanguage(language: string | null | undefined): string | null {
  if (language === null || language === undefined) return null;
  const code = language.trim().replaceAll("_", "-").split("-", 1)[0]!.toLowerCase();
  return code || null;
}

/**
 * `capitalisation` when the caller did not set it: off in French, on otherwise. Measured on 61
 * French transcriptions: the transcriber's case is already right, and every model only made it
 * worse.
 */
export function defaultCapitalisation(language: string | null | undefined): boolean {
  return normaliseLanguage(language) !== "fr";
}

/**
 * From feature names — `"punctuation,grammar"` or a list; every switch set, unknown names refused.
 */
export function featuresFromNames(names: string | readonly string[]): CorrectionFeatures {
  const wanted = (typeof names === "string" ? names.split(",") : names.map(String)).map((n) => n.trim());
  const unknown = wanted.filter((n) => n && !(FEATURES as readonly string[]).includes(n));
  if (unknown.length > 0) throw new ConfigurationError(`unknown feature(s) ${unknown.join(", ")}; known: ${[...FEATURES].sort().join(", ")}`);
  const chosen = new Set(wanted.filter(Boolean));
  return Object.fromEntries(FEATURES.map((f) => [f, chosen.has(f)])) as CorrectionFeatures;
}

export function enabledFeatures(features: CorrectionFeatures): Set<Feature> {
  return new Set(FEATURES.filter((f) => features[f]));
}

/** How a text is cut into pieces. No overlap: two corrections of one span cannot be spliced. */
export interface SegmentationConfig {
  splitOnSentences: boolean;
  maxWords: number;
  /** Pieces shorter than this are folded into a neighbour. */
  minChars: number;
  /** How much of the window's tail may be searched backwards for a better cut. */
  cutSearchRatio: number;
}

export const DEFAULT_SEGMENTATION: SegmentationConfig = { splitOnSentences: true, maxWords: 64, minChars: 2, cutSearchRatio: 0.25 };

/** Decoding parameters and the length guard, for the generative stages. */
export interface RewriteConfig {
  strategy: "beam" | "greedy";
  beamSize: number;
  lengthPenalty: number;
  repetitionPenalty: number;
  /** Ceiling on the generated tokens, ON TOP of the input's own length. */
  maxNewTokens: number;
  minLengthRatio: number;
  maxLengthRatio: number;
  onLengthViolation: LengthGuardAction;
}

export const DEFAULT_REWRITE: RewriteConfig = {
  strategy: "beam",
  beamSize: 4,
  lengthPenalty: 1,
  repetitionPenalty: 1,
  maxNewTokens: 32,
  minLengthRatio: 0.6,
  maxLengthRatio: 1.6,
  onLengthViolation: "keep_original",
};

export function rewriteConfig(overrides: Partial<RewriteConfig> = {}): RewriteConfig {
  const config = { ...DEFAULT_REWRITE, ...overrides };
  if (config.beamSize < 1) throw new ConfigurationError(`beam_size must be at least 1, got ${config.beamSize}`);
  if (config.minLengthRatio > config.maxLengthRatio) {
    throw new ConfigurationError(
      `min_length_ratio=${config.minLengthRatio} is above max_length_ratio=${config.maxLengthRatio}: no output can pass the guard, so every piece would fall back to its original.`,
    );
  }
  return config;
}

/** Whether a rewrite is a plausible correction of its input, in characters. */
export function withinBounds(config: RewriteConfig, originalChars: number, rewrittenChars: number): boolean {
  if (originalChars <= 0) return true;
  const ratio = rewrittenChars / originalChars;
  return config.minLengthRatio <= ratio && ratio <= config.maxLengthRatio;
}

/** What one stage did, over the whole pool of a call. */
export interface StageReport {
  stage: string;
  model: string;
  features: string[];
  pieces: number;
  batches: number;
  /** Pieces this stage actually altered. */
  changed: number;
  /** Pieces the length guard sent back to their original. */
  reverted: number;
  elapsedS: number;
}

export interface CorrectionResult {
  text: string;
  original: string;
  features: string[];
  stages: StageReport[];
  pieces: number;
  /** The models that ran, in order, `+`-joined. */
  modelName: string;
  /** The language the corrector was given (nothing is detected). */
  language: string | null;
  elapsedS: number;
  metadata: Record<string, unknown>;
}
