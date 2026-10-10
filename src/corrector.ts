/** TextCorrector — the orchestrator. The port of `ynonnx_correct.TextCorrector`. */

import { huggingFaceResolver, loadModel, type LoadOptions, type ModelResolver } from "@youni/onnxweb-core";
import {
  ConfigurationError,
  IndexedDbStore,
  ModelNotFoundError,
  ResultCache,
  ResumableRun,
  SUBTITLE_EXTENSIONS,
  Tokenizer,
  defaultLogger,
  mapCues,
  splitFrontMatter,
  type KeyValueStore,
  type Logger,
  type TensorFactory,
} from "@youni/onnxweb-core/core";

import { DEFAULT_MAX_BATCH_ITEMS, DEFAULT_TOKEN_BUDGET, MODEL_CATALOGUE, PRESETS } from "./catalogue.js";
import { buildPlan, describePlan, type StagePlan } from "./plan.js";
import { CorrectRegistry, resolveModel } from "./registry.js";
import { Segmenter, type Span } from "./segmenter.js";
import { JsonVocabulary, RewriteStage, SentencePieceVocabulary, TaggingStage, readLabelMap, type CorrectionStage, type Vocabulary } from "./stages.js";
import { applyLengthGuard, correctMarkdown, enforceVerbatim, lengthBatches } from "./text.js";
import {
  defaultCapitalisation,
  enabledFeatures,
  featuresFromNames,
  normaliseLanguage,
  rewriteConfig,
  YnONNXCorrectError,
  type CorrectionFeatures,
  type CorrectionResult,
  type Prefer,
  type RewriteConfig,
  type SegmentationConfig,
  type StageReport,
} from "./types.js";

export interface TextCorrectorOptions extends Omit<LoadOptions, "model" | "computeType" | "resolver"> {
  Tensor: TensorFactory;
  punctuation?: boolean;
  /** Unset: off in French, on otherwise (measured). */
  capitalisation?: boolean | null;
  /** REWRITES words; off by default. */
  spelling?: boolean;
  /** REWRITES words; off by default. */
  grammar?: boolean;
  punctuationModel?: string;
  rewriteModel?: string;
  /** Steers `auto` on both models; nothing is detected. */
  language?: string | null;
  prefer?: Prefer;
  /**
   * As published by default (the float32 graphs Python reads); `int8` where a repository has one.
   */
  computeType?: string;
  /** `auto` for a token budget, or a fixed piece count. */
  batchSize?: number | "auto";
  segmentation?: Partial<SegmentationConfig>;
  rewriting?: Partial<RewriteConfig>;
  maxBatchItems?: number;
  endpoint?: string;
  /** Resolvers by catalogue key, for the models without a published ONNX (or any other). */
  resolvers?: Readonly<Record<string, ModelResolver>>;
  logger?: Logger;
  checkpoint?: KeyValueStore | boolean;
  cache?: KeyValueStore | boolean;
}

/** Texts corrected together in one pool (and checkpointed together). */
export const POOL_TEXTS = 256;

const round4 = (x: number) => Math.round(x * 10000) / 10000;

export class TextCorrector {
  readonly features: CorrectionFeatures;
  readonly plans: readonly StagePlan[];
  readonly language: string | null;
  readonly segmenter: Segmenter;
  readonly rewriting: RewriteConfig;
  readonly registry: CorrectRegistry;
  readonly #logger: Logger;
  readonly #stages = new Map<string, CorrectionStage>();
  readonly #checkpoint: KeyValueStore | null;
  readonly #cache: ResultCache | null;

  constructor(readonly options: TextCorrectorOptions) {
    this.#logger = options.logger ?? defaultLogger;

    this.language = normaliseLanguage(options.language ?? null);
    this.features = {
      punctuation: options.punctuation ?? true,
      capitalisation: options.capitalisation ?? defaultCapitalisation(this.language),
      spelling: options.spelling ?? false,
      grammar: options.grammar ?? false,
    };
    const resolvers = options.resolvers ?? {};
    this.registry = new CorrectRegistry(this.#logger, (key) => Boolean(MODEL_CATALOGUE[key]?.onnxRepo) || key in resolvers);
    for (const name of [options.punctuationModel ?? "auto", options.rewriteModel ?? "auto"]) this.registry.requireKnown(name);
    this.plans = buildPlan(this.features, this.registry, {
      punctuationModel: options.punctuationModel ?? "auto",
      rewriteModel: options.rewriteModel ?? "auto",
      language: this.language,
      prefer: options.prefer ?? "balanced",
    });
    this.segmenter = new Segmenter(options.segmentation ?? {});
    this.rewriting = rewriteConfig(options.rewriting ?? {});
    this.#checkpoint = options.checkpoint === true ? new IndexedDbStore() : options.checkpoint || null;
    const cache = options.cache === true ? new IndexedDbStore() : options.cache || null;
    this.#cache = cache ? new ResultCache(cache, "ynonnx-correct") : null;
    this.#logger.debug?.(`plan: ${describePlan(this.plans)}`);
  }

  /** From a named preset — `transcript`, `prose`, `grammar-only`, `cpu`. */
  static fromPreset(name: string, options: TextCorrectorOptions): TextCorrector {
    const preset = PRESETS[name];
    if (!preset) throw new ConfigurationError(`Unknown preset '${name}'. Known: ${Object.keys(PRESETS).sort().join(", ")}`);
    const switches = featuresFromNames(preset.features);
    return new TextCorrector({
      ...(preset.prefer ? { prefer: preset.prefer } : {}),
      ...(preset.computeType ? { computeType: preset.computeType } : {}),
      ...(name === "cpu" ? { backends: "wasm" as never } : {}),
      ...switches,

      capitalisation: switches.capitalisation ? null : false,
      ...options,
    });
  }

  /** Punctuation and case only, verbatim guaranteed: no option can switch a rewrite on. */
  static forTranscription(options: TextCorrectorOptions): TextCorrector {
    const { spelling: _s, grammar: _g, ...rest } = options;
    return new TextCorrector({ ...rest, punctuation: true, spelling: false, grammar: false });
  }

  get rewritesWords(): boolean {
    return this.plans.some((p) => p.kind === "rewrite");
  }

  get modelName(): string {
    return this.plans.map((p) => p.model).join("+");
  }

  get isLoaded(): boolean {
    return this.plans.every((p) => this.#stages.has(p.model));
  }

  getModelInfo(): Record<string, unknown> {
    return {
      modelName: this.modelName,
      features: [...enabledFeatures(this.features)].sort(),
      language: this.language,
      computeType: this.options.computeType ?? "float32",
      loaded: this.isLoaded,
      stages: this.plans.map((plan) => {
        const entry = this.registry.entry(plan.model);
        return {
          name: plan.model,
          hfId: entry.hfId,
          family: entry.family,
          kind: plan.kind,
          features: [...plan.features].sort(),
          maxLength: entry.maxLength,
          tokenizerKind: entry.tokenizerKind,
          languages: [...entry.languages],
          paramsM: entry.paramsM,
          licence: entry.licence,
          backend: this.#stages.get(plan.model)?.backend ?? null,
          notes: entry.notes,
        };
      }),
    };
  }

  async #ensureStage(plan: StagePlan): Promise<CorrectionStage> {
    const existing = this.#stages.get(plan.model);
    if (existing) return existing;
    const stage = await this.#buildStage(plan);
    this.#stages.set(plan.model, stage);
    this.#logger.info?.(`stage ready | kind=${plan.kind} model='${plan.model}' features=${[...plan.features].sort().join(",")} backend=${stage.backend}`);
    return stage;
  }

  async #buildStage(plan: StagePlan): Promise<CorrectionStage> {
    const key = resolveModel(plan.model);
    const entry = this.registry.entry(plan.model);
    const { Tensor, endpoint, resolvers, logger, computeType, checkpoint: _c, cache: _k, punctuation: _p, capitalisation: _ca, spelling: _s, grammar: _g, punctuationModel: _pm, rewriteModel: _rm, language: _l, prefer: _pr, batchSize: _b, segmentation: _se, rewriting: _rw, maxBatchItems: _m, ...rest } = this.options;
    const repo = entry.onnxRepo ?? (plan.model.includes("/") ? plan.model : null);
    const resolver = resolvers?.[key] ?? (repo ? huggingFaceResolver({ repo, ...(entry.onnxSubfolder ? { subfolder: entry.onnxSubfolder } : {}), manifest: false, ...(endpoint ? { endpoint } : {}) }) : null);
    if (!resolver) throw new ModelNotFoundError(`${plan.model} has no published ONNX; pass resolvers: { '${key}': <a resolver that serves an export of ${entry.hfId}> }`);
    const bundle = await loadModel({
      extraFiles: [entry.tokenizerFile, "tokenizer_config.json", "config.json", "config.yaml", "generation_config.json"],
      ...rest,
      ...(logger ? { logger } : {}),
      resolver,
      model: { name: repo ?? key, computeType: computeType ?? "float32" },
    });
    let vocabulary: Vocabulary;
    if (entry.tokenizerKind === "sentencepiece") {
      const bytes = bundle.asset(entry.tokenizerFile);
      if (!bytes) throw new YnONNXCorrectError(`${plan.model} came without its SentencePiece vocabulary ${entry.tokenizerFile}`);
      vocabulary = new SentencePieceVocabulary(bytes, this.registry.isLowercaseModel(plan.model));
    } else {
      const json = bundle.assetText(entry.tokenizerFile) ?? bundle.assetText("tokenizer.json");
      if (!json) throw new YnONNXCorrectError(`${plan.model} came without a tokenizer.json`);
      vocabulary = new JsonVocabulary(Tokenizer.fromJson(json, bundle.assetText("tokenizer_config.json")));
    }
    if (plan.kind === "rewrite") return new RewriteStage(bundle, vocabulary, Tensor, plan.model, entry.family, entry.maxLength, this.rewriting);
    const heads = entry.heads ?? ["post_punctuation"];
    const labels = readLabelMap({ configJson: bundle.assetText("config.json"), configYaml: bundle.assetText("config.yaml") }, heads);
    const stage = new TaggingStage(bundle, bundle.only, vocabulary, Tensor, heads, labels, plan.model, entry.maxLength, plan.features);
    stage.verify();
    return stage;
  }

  /** Load every stage now rather than on the first call. */
  async loadModel(): Promise<void> {
    for (const plan of this.plans) await this.#ensureStage(plan);
  }

  async unloadModel(): Promise<void> {
    for (const stage of this.#stages.values()) await stage.close();
    this.#stages.clear();
  }

  async close(): Promise<void> {
    await this.unloadModel();
  }

  /** Correct one text, or a list, pooling every piece of all of them. */
  async correct(texts: string, options?: { batchSize?: number; signal?: AbortSignal }): Promise<CorrectionResult>;
  async correct(texts: readonly string[], options?: { batchSize?: number; signal?: AbortSignal }): Promise<CorrectionResult[]>;
  async correct(texts: string | readonly string[], options: { batchSize?: number; signal?: AbortSignal } = {}): Promise<CorrectionResult | CorrectionResult[]> {
    const single = typeof texts === "string";
    const items = single ? [texts] : [...texts];
    if (items.length === 0) return [];
    const run = await ResumableRun.open({
      checkpoint: this.#checkpoint !== null,
      ...(this.#checkpoint ? { store: this.#checkpoint } : {}),
      inputs: items,
      settings: {
        models: this.plans.map((p) => p.model),
        features: [...enabledFeatures(this.features)].sort(),
        language: this.language,
        computeType: this.options.computeType ?? "float32",
        rewriting: this.rewriting,
        segmentation: this.segmenter.config,
      },
      cache: this.#cache,
    });
    const missing: number[] = [];
    for (let i = 0; i < items.length; i += 1) if (!(await run.has(i))) missing.push(i);
    for (let start = 0; start < missing.length; start += POOL_TEXTS) {
      const block = missing.slice(start, start + POOL_TEXTS);
      const results = await this.#correctPool(block.map((i) => items[i]!), options.batchSize ?? null, options.signal);
      for (const [n, index] of block.entries()) await run.record(index, results[n]);
    }
    const results = items.map((_, i) => run.get(i) as CorrectionResult);
    await run.complete();
    return single ? results[0]! : results;
  }

  /**
   * Correct a file, its structure kept: `.srt`/`.vtt` (cues corrected, numbering and timings kept,
   * every cue in one pool), `.md` (front matter and code kept), anything else as plain text.
   */
  async correctFile(file: (Blob & { name: string }) | { name: string; text: string }, options: { batchSize?: number; signal?: AbortSignal } = {}): Promise<CorrectionResult> {
    const started = performance.now();
    const original = "text" in file && typeof file.text === "string" ? file.text : await (file as Blob).text();
    const dot = file.name.lastIndexOf(".");
    const suffix = dot >= 0 ? file.name.slice(dot).toLowerCase() : "";
    const reports: StageReport[] = [];
    let pieces = 0;
    const fix = async (texts: string[]) => {
      const results = await this.correct(texts, options);
      if (results.length > 0) reports.push(...results[0]!.stages);
      pieces += results.reduce((s, r) => s + r.pieces, 0);
      return results.map((r) => r.text);
    };
    let text: string;
    let metadata: Record<string, unknown>;
    if (SUBTITLE_EXTENSIONS.has(suffix)) {
      const [mapped, cues] = await mapCues(original, fix);
      text = mapped;
      metadata = { format: suffix.slice(1), cues };
    } else if (suffix === ".md") {
      const [body, front] = splitFrontMatter(original);
      text = original.slice(0, original.length - body.length) + (await correctMarkdown(body, fix));
      metadata = { format: "md", ...front };
    } else {
      text = (await fix([original]))[0]!;
      metadata = { format: suffix.slice(1) || "txt" };
    }
    return {
      text,
      original,
      features: [...enabledFeatures(this.features)].sort(),
      stages: reports,
      pieces,
      modelName: this.modelName,
      language: this.language,
      elapsedS: round4((performance.now() - started) / 1000),
      metadata,
    };
  }

  async #correctPool(texts: string[], batchSize: number | null, signal?: AbortSignal): Promise<CorrectionResult[]> {
    const started = performance.now();
    const spans: Span[][] = texts.map((t) => this.segmenter.spans(t));
    const owners: number[] = [];
    let current: string[] = [];
    texts.forEach((text, owner) => {
      const chars = Array.from(text);
      for (const [s, e] of spans[owner]!) {
        current.push(chars.slice(s, e).join(""));
        owners.push(owner);
      }
    });
    const reports: StageReport[] = [];
    for (const plan of this.plans) {
      let report: StageReport;
      [current, report] = await this.#runStage(plan, current, batchSize, signal);
      reports.push(report);
    }
    const elapsed = round4((performance.now() - started) / 1000);
    return texts.map((text, owner) => {
      const replacements = current.filter((_, i) => owners[i] === owner);
      const own = spans[owner]!;
      return {
        text: own.length > 0 ? this.segmenter.splice(text, own, replacements) : text,
        original: text,
        features: [...enabledFeatures(this.features)].sort(),
        stages: reports,
        pieces: own.length,
        modelName: this.modelName,
        language: this.language,
        elapsedS: elapsed,
        metadata: {},
      };
    });
  }

  async #runStage(plan: StagePlan, pieces: string[], batchSize: number | null, signal?: AbortSignal): Promise<[string[], StageReport]> {
    const stage = await this.#ensureStage(plan);
    const started = performance.now();
    const report: StageReport = { stage: plan.kind, model: plan.model, features: [...plan.features].sort(), pieces: pieces.length, batches: 0, changed: 0, reverted: 0, elapsedS: 0 };
    if (pieces.length === 0) return [pieces, report];
    const lengths = stage.tokenLengths(pieces);
    const fixed = batchSize ?? (typeof this.options.batchSize === "number" ? this.options.batchSize : null);
    let out = new Array<string>(pieces.length).fill("");
    for (const indices of lengthBatches(lengths, this.#tokenBudget(plan), { batchSize: fixed, maxBatchItems: this.options.maxBatchItems ?? DEFAULT_MAX_BATCH_ITEMS })) {
      signal?.throwIfAborted();
      const corrected = await stage.apply(indices.map((i) => pieces[i]!), signal);
      if (corrected.length !== indices.length) throw new YnONNXCorrectError(`${plan.model} returned ${corrected.length} pieces for ${indices.length}; the splice is by position`);
      indices.forEach((index, k) => (out[index] = corrected[k]!));
      report.batches += 1;
    }

    if (stage.kind === "rewrite") [out, report.reverted] = applyLengthGuard(pieces, out, this.rewriting, this.#logger);
    else enforceVerbatim(pieces, out, plan.model, plan.kind);
    report.changed = pieces.filter((p, i) => p !== out[i]).length;
    report.elapsedS = round4((performance.now() - started) / 1000);
    return [out, report];
  }

  /** Subwords per batch; divided by the beam width on a rewrite stage. */
  #tokenBudget(plan: StagePlan): number {
    return plan.kind === "rewrite" ? Math.max(Math.floor(DEFAULT_TOKEN_BUDGET / Math.max(this.rewriting.beamSize, 1)), 256) : DEFAULT_TOKEN_BUDGET;
  }
}
