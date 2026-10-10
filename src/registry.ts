/** Reading the catalogue, and choosing for `auto`. The port of `ynonnx_correct.models.registry`. */

import { ConfigurationError, defaultLogger, type Logger } from "@youni/onnxweb-core/core";

import { HF_ID_TO_ALIAS, MODEL_ALIASES, MODEL_CATALOGUE, PUNCT_AND_CASE, type CorrectionEntry } from "./catalogue.js";
import { normaliseLanguage, type Feature, type Prefer, type StageKind } from "./types.js";

/** What an unknown model gets: a Hugging Face token classifier, one punctuation head. */
export const NEUTRAL_DEFAULTS: CorrectionEntry = {
  hfId: "",
  onnxRepo: null,
  onnxSubfolder: "onnx",
  graphFile: "model.onnx",
  tokenizerFile: "tokenizer.json",
  tokenizerKind: "tokenizers",
  family: "unknown",
  kind: "tagging",
  features: PUNCT_AND_CASE,
  maxLength: 512,
  heads: ["post_punctuation"],
  languages: [],
  paramsM: 0,
  licence: "",
  notes: null,
};

/** Vocabulary files whose name says the model was trained on lowercased input. */
const LOWERCASE_MARKERS = ["_lc", "lowercase"];
const unknownLanguage = (language: string | null) => language === null || language === "auto";

/** Alias, `org/repo` or path → the catalogue key, or the name itself. */
export function resolveModel(name: string): string {
  const key = name.trim();
  if (key in MODEL_CATALOGUE) return key;
  return MODEL_ALIASES[key] ?? MODEL_ALIASES[key.toLowerCase()] ?? key;
}

/** The catalogue row for a model, or neutral defaults. */
export function entryFor(name: string): CorrectionEntry {
  const key = resolveModel(name);
  return MODEL_CATALOGUE[key] ?? { ...NEUTRAL_DEFAULTS, hfId: name };
}

export function isKnown(name: string): boolean {
  return resolveModel(name) in MODEL_CATALOGUE || name in HF_ID_TO_ALIAS;
}

export class CorrectRegistry {
  readonly #warned = new Set<string>();

  /**
   * `servable(key)`: whether a catalogued model can be fetched here — by default, when it has a
   * published ONNX.
   */
  constructor(
    readonly logger: Logger = defaultLogger,
    readonly servable: (key: string) => boolean = (key) => Boolean(MODEL_CATALOGUE[key]?.onnxRepo),
  ) {}

  entry(name: string): CorrectionEntry {
    if (!isKnown(name) && !this.#warned.has(name)) {
      this.#warned.add(name);
      this.logger.warn(
        `'${name}' is not in the catalogue: assuming a Hugging Face token classifier with one punctuation head, a tokenizer.json and a 512-token window.`,
      );
    }
    return entryFor(name);
  }

  kindOf(name: string): StageKind {
    return this.entry(name).kind;
  }

  featuresOf(name: string): Set<Feature> {
    return new Set(this.entry(name).features);
  }

  languagesOf(name: string): readonly string[] {
    return this.entry(name).languages;
  }

  /** Whether a model's row claims `language`; an unknown language is claimed by all. */
  declares(name: string, language: string | null): boolean {
    const code = normaliseLanguage(language);
    if (unknownLanguage(code)) return true;
    return this.languagesOf(name).includes(code!);
  }

  /** Warn once when a model named explicitly does not declare `language`. */
  warnIfUndeclared(name: string, language: string | null): void {
    if (unknownLanguage(language) || !isKnown(name) || this.declares(name, language)) return;
    const seen = `${name}|${language}`;
    if (this.#warned.has(seen)) return;
    this.#warned.add(seen);
    const declared = this.languagesOf(name).join(", ") || "none confirmed";
    this.logger.warn(
      `'${name}' was named explicitly for language='${language}', which its model card does not declare (declared: ${declared}). It is used as asked; pass "auto" instead to get a model that declares '${language}', or a refusal if none does.`,
    );
  }

  /** Whether this checkpoint expects lowercased input (read from its vocabulary's name). */
  isLowercaseModel(name: string): boolean {
    const file = this.entry(name).tokenizerFile.toLowerCase();
    return LOWERCASE_MARKERS.some((marker) => file.includes(marker));
  }

  /** Refuse a name that is neither catalogued nor plausibly a repository or a path. */
  requireKnown(name: string): void {
    if (name === "auto" || isKnown(name) || name.includes("/") || name.includes("\\")) return;
    const known = [...Object.keys(MODEL_CATALOGUE), ...Object.keys(MODEL_ALIASES).filter((a) => !a.includes("/"))];
    const close = known.filter((k) => k.startsWith(name.slice(0, 3).toLowerCase())).slice(0, 3);
    throw new ConfigurationError(`Unknown model '${name}'.${close.length ? ` Did you mean ${close.join(", ")}?` : ""} Known: ${Object.keys(MODEL_CATALOGUE).join(", ")}`);
  }

  #first(candidates: readonly string[], language: string | null): string | null {
    return candidates.find((key) => this.servable(key) && this.declares(key, language)) ?? null;
  }

  /** A punctuation model for `auto`, or null when none declares `language`. */
  resolvePunctuation(language: string | null = null, prefer: Prefer = "balanced"): string | null {
    const code = normaliseLanguage(language);
    if (code === "fr") {
      return this.#first(prefer === "quality" ? ["fullstop-large"] : ["punctuate-all", "xlm-punct-cap-seg"], language);
    }
    let candidates: string[];
    if (language === "en" && prefer === "fast") candidates = ["punct-cap-seg-en"];
    else if (code !== null && ["es", "ca", "pt", "it", "ro"].includes(code)) candidates = ["punct-cap-seg-romance"];
    else if (prefer === "fast") candidates = ["punct-cap-seg-47", "punctuate-all"];
    else candidates = ["xlm-punct-cap-seg", "punctuate-all"];
    return this.#first(candidates, language);
  }

  /** A spelling/grammar model for `auto`, or null when none declares `language`. */
  resolveRewrite(features: ReadonlySet<Feature>, language: string | null = null): string | null {
    if (unknownLanguage(language) || language === "en") {
      return features.size === 1 && features.has("spelling") ? "spelling-en" : "grammar-synthesis-small";
    }
    const declared = Object.entries(MODEL_CATALOGUE)
      .filter(([key, entry]) => entry.kind === "rewrite" && this.servable(key) && this.declares(key, language) && entry.features.some((f) => features.has(f)))
      .map(([key]) => key);
    const complete = declared.filter((key) => [...features].every((f) => this.featuresOf(key).has(f)));
    return (complete.length > 0 ? complete : declared)[0] ?? null;
  }

  /** Catalogue keys matching simple filters. */
  listModels(filters: { feature?: Feature; kind?: StageKind; language?: string; maxParamsM?: number; servableOnly?: boolean } = {}): string[] {
    return Object.entries(MODEL_CATALOGUE)
      .filter(([key, entry]) => {
        if (filters.feature && !entry.features.includes(filters.feature)) return false;
        if (filters.kind && entry.kind !== filters.kind) return false;
        if (filters.language && !entry.languages.includes(filters.language)) return false;
        if (filters.maxParamsM !== undefined && entry.paramsM > filters.maxParamsM) return false;
        if (filters.servableOnly && !this.servable(key)) return false;
        return true;
      })
      .map(([key]) => key);
  }
}
