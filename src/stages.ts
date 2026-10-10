/** The vocabularies and the two stages. Ports of `ynonnx_correct.stages`. */

import type { ModelBundle, ModelSession } from "@youni/onnxweb-core";
import { GenerationEngine, SentencePieceProcessor, type TensorFactory, type Tokenizer } from "@youni/onnxweb-core/core";

import { applyLabels, capitaliseSentences, NO_MARK } from "./text.js";
import { StageMismatchError, type Feature, type RewriteConfig, type StageKind } from "./types.js";

/** Text in, ids out, and where each id came from in the text. */
export interface Vocabulary {
  /** Labels are per word (a Hugging Face token classifier) rather than per subword. */
  readonly wordLevel: boolean;
  encode(text: string): number[];
  encodeWithOffsets(text: string): [number[], [number, number][]];
  decode(ids: readonly number[]): string;
  /** Per piece, the characters before its span (SentencePiece's `▁`). */
  leads(text: string): number[];
  readonly bosId: number | null;
  readonly eosId: number | null;
  readonly padId: number;
}

const firstId = (tokenizer: Tokenizer, names: readonly string[]) => {
  for (const name of names) {
    const found = tokenizer.tokenToId(name);
    if (found !== null && found !== undefined) return found;
  }
  return null;
};

/** A `tokenizer.json`, read by the core's tokenizer. */
export class JsonVocabulary implements Vocabulary {
  readonly wordLevel = true;
  constructor(readonly tokenizer: Tokenizer) {}

  encode(text: string): number[] {
    return this.tokenizer.encode(text, { addSpecialTokens: false });
  }

  encodeWithOffsets(text: string): [number[], [number, number][]] {
    const encoded = this.tokenizer.encodeWithOffsets(text, { addSpecialTokens: false });
    return [encoded.ids, encoded.offsets];
  }

  decode(ids: readonly number[]): string {
    return this.tokenizer.decode(ids, { skipSpecialTokens: true });
  }

  leads(text: string): number[] {
    return new Array<number>(this.encode(text).length).fill(0);
  }

  get bosId(): number | null {
    return firstId(this.tokenizer, ["<s>", "[CLS]", "<|startoftext|>"]);
  }

  get eosId(): number | null {
    return firstId(this.tokenizer, ["</s>", "[SEP]", "<|endoftext|>"]);
  }

  get padId(): number {
    return firstId(this.tokenizer, ["<pad>", "[PAD]", "<|endoftext|>"]) ?? 0;
  }
}

const SPACE_MARKER = "▁";

/**
 * A raw SentencePiece `.model`, spans recovered by walking the pieces against the source.
 * `lowercase` for checkpoints trained on lowercased input.
 */
export class SentencePieceVocabulary implements Vocabulary {
  readonly wordLevel = false;
  readonly processor: SentencePieceProcessor;

  constructor(
    bytes: Uint8Array | ArrayBuffer,
    readonly lowercase = false,
  ) {
    this.processor = new SentencePieceProcessor(bytes);
  }

  #prepared(text: string): string {
    return this.lowercase ? text.toLowerCase() : text;
  }

  encode(text: string): number[] {
    return this.processor.encodeIds(this.#prepared(text));
  }

  /**
   * Ids, and each piece's span in the ORIGINAL (cased) text; a piece not found gets an empty span.
   */
  encodeWithOffsets(text: string): [number[], [number, number][]] {
    const prepared = this.#prepared(text);
    const ids = this.processor.encodeIds(prepared);
    const pieces = this.processor.encodePieces(prepared);
    const haystack = text.toLowerCase();
    const offsets: [number, number][] = [];
    let cursor = 0;
    for (const piece of pieces) {
      const surface = piece.replaceAll(SPACE_MARKER, "");
      if (!surface) {
        offsets.push([cursor, cursor]);
        continue;
      }
      const found = haystack.indexOf(surface.toLowerCase(), cursor);
      if (found < 0) {
        offsets.push([cursor, cursor]);
        continue;
      }
      offsets.push([found, found + surface.length]);
      cursor = found + surface.length;
    }
    return [ids, offsets];
  }

  decode(ids: readonly number[]): string {
    return ids
      .map((id) => this.processor.model.pieces[id]?.piece ?? "")
      .join("")
      .replaceAll(SPACE_MARKER, " ")
      .trim();
  }

  leads(text: string): number[] {
    return this.processor.encodePieces(this.#prepared(text)).map((p) => Array.from(p).length - Array.from(p.replace(/^▁+/u, "")).length);
  }

  get bosId(): number | null {
    return this.processor.bosId;
  }

  get eosId(): number | null {
    return this.processor.eosId;
  }

  get padId(): number {
    return this.processor.padId;
  }
}

/** Head roles the tagging stage acts on; `segmentation` changes no character. */
export const ACTED_ON = ["pre_punctuation", "post_punctuation", "capitalisation"] as const;
/** Ways a label set spells "nothing to add here". */
const NULL_LABELS = new Set(["", "0", "O", "NONE", "none", "null", "<NULL>", "_"]);

export function normalizeLabel(label: unknown): string {
  const text = String(label).trim();
  return NULL_LABELS.has(text) ? NO_MARK : text;
}

/** Label lists per head, from a `config.json` (`id2label`) or a NeMo `config.yaml`. */
export function readLabelMap(files: { configJson?: string | null; configYaml?: string | null }, heads: readonly string[]): Record<string, string[]> {
  if (files.configJson) {
    const id2label = (JSON.parse(files.configJson) as { id2label?: Record<string, string> }).id2label ?? {};
    const keys = Object.keys(id2label).sort((a, b) => Number(a) - Number(b));
    if (keys.length > 0) {
      const role = heads.includes("post_punctuation") ? "post_punctuation" : heads[0]!;
      return { [role]: keys.map((k) => normalizeLabel(id2label[k])) };
    }
  }
  if (files.configYaml) {
    const found: Record<string, string[]> = {};
    for (const [role, key] of [["pre_punctuation", "pre_labels"], ["post_punctuation", "post_labels"]] as const) {
      const values = yamlList(files.configYaml, key);
      if (values && values.length > 0 && heads.includes(role)) found[role] = values.map(normalizeLabel);
    }

    if (heads.includes("capitalisation")) found["capitalisation"] = ["", "upper"];
    return found;
  }
  throw new StageMismatchError(
    "no label map beside the graph: looked for config.json (id2label) and config.yaml (pre_labels/post_labels). The mapping from an output index to a mark belongs to the checkpoint, and no default is right for two of them.",
  );
}

/** The items of a flow list's inside, split on commas OUTSIDE quotes (`".", ","` is two items). */
function flowItems(inside: string): string[] {
  const items: string[] = [];
  let current = "";
  let quote: string | null = null;
  for (let i = 0; i < inside.length; i += 1) {
    const c = inside[i]!;
    if (quote) {
      current += c;
      if (c === "\\" && quote === '"' && i + 1 < inside.length) current += inside[++i]!;
      else if (c === quote) {
        if (quote === "'" && inside[i + 1] === "'") current += inside[++i]!;
        else quote = null;
      }
    } else if (c === '"' || c === "'") {
      quote = c;
      current += c;
    } else if (c === ",") {
      items.push(current);
      current = "";
    } else current += c;
  }
  items.push(current);
  return items;
}

/** A top-level YAML list (`key:` then `- item` lines, or `key: [a, b]`), scalars only. */
export function yamlList(yaml: string, key: string): string[] | null {
  const lines = yaml.split(/\r?\n/);
  const scalar = (raw: string) => {
    const text = raw.trim();
    if (text.length >= 2 && text.startsWith("'") && text.endsWith("'")) return text.slice(1, -1).replaceAll("''", "'");
    if (text.length >= 2 && text.startsWith('"') && text.endsWith('"')) return JSON.parse(text) as string;
    return text;
  };
  for (const [i, line] of lines.entries()) {
    const match = new RegExp(`^${key}\\s*:\\s*(.*)$`).exec(line);
    if (!match) continue;
    let inline = match[1]!.trim();
    if (inline.startsWith("[")) {
      for (const next of lines.slice(i + 1)) {
        if (inline.includes("]")) break;
        inline += ` ${next.trim()}`;
      }
      return flowItems(inline.slice(1, inline.lastIndexOf("]"))).map(scalar).filter((v) => v !== "");
    }
    const values: string[] = [];
    for (const next of lines.slice(i + 1)) {
      const item = /^\s*-\s?(.*)$/.exec(next);
      if (item) values.push(scalar(item[1]!));
      else if (next.trim() && !next.trim().startsWith("#")) break;
    }
    return values;
  }
  return null;
}

export interface CorrectionStage {
  readonly kind: StageKind;
  readonly modelName: string;
  readonly backend: string;
  apply(pieces: string[], signal?: AbortSignal): Promise<string[]>;
  tokenLengths(pieces: string[]): number[];
  close(): Promise<void>;
}

interface OrtTensor {
  data: ArrayLike<number | bigint | boolean>;
  dims: readonly number[];
  type: string;
}

export class TaggingStage implements CorrectionStage {
  readonly kind = "tagging" as const;

  constructor(
    readonly bundle: ModelBundle,
    readonly session: ModelSession,
    readonly vocabulary: Vocabulary,
    readonly Tensor: TensorFactory,
    readonly heads: readonly string[],
    readonly labels: Record<string, string[]>,
    readonly modelName: string,
    readonly maxLength: number,
    readonly wants: ReadonlySet<Feature>,
  ) {}

  get backend(): string {
    return this.session.backend;
  }

  /** The graph checked against what the catalogue promised. */
  verify(): void {
    const outputs = this.session.outputs.length;
    if (outputs !== this.heads.length) {
      throw new StageMismatchError(
        `${this.modelName} declares heads ${JSON.stringify(this.heads)} (${this.heads.length}), but the graph emits ${outputs} output(s). Every head after the first mismatch would be read as its neighbour. Fix the catalogue entry, or re-export the model.`,
      );
    }
    for (const role of this.heads) {
      if ((ACTED_ON as readonly string[]).includes(role) && !(role in this.labels)) {
        throw new StageMismatchError(`no label map for the '${role}' head of ${this.modelName}; guessing it puts the wrong mark everywhere without error.`);
      }
    }
  }

  tokenLengths(pieces: string[]): number[] {
    return pieces.map((p) => Math.min(this.vocabulary.encode(p).length, this.maxLength));
  }

  get #wraps(): boolean {
    return this.vocabulary.bosId !== null && this.vocabulary.eosId !== null;
  }

  async apply(pieces: string[]): Promise<string[]> {
    if (pieces.length === 0) return [];
    const room = this.maxLength - (this.#wraps ? 2 : 0);
    const encoded = pieces.map((p) => this.vocabulary.encodeWithOffsets(p));
    const ids = encoded.map(([row]) => row.slice(0, room));
    const offsets = encoded.map(([, row]) => row.slice(0, room));
    const leads = pieces.map((p) => this.vocabulary.leads(p).slice(0, room));
    const heads = await this.#run(ids);
    return pieces.map((piece, row) => this.#assemble(piece, offsets[row]!, heads, row, leads[row]!));
  }

  async #run(rows: number[][]): Promise<Map<string, OrtTensor>> {
    const wrapped = this.#wraps ? rows.map((r) => [this.vocabulary.bosId!, ...r, this.vocabulary.eosId!]) : rows;
    const width = Math.max(0, ...wrapped.map((r) => r.length));
    const ids = new BigInt64Array(wrapped.length * width).fill(BigInt(this.vocabulary.padId));
    const mask = new BigInt64Array(wrapped.length * width);
    wrapped.forEach((r, i) => r.forEach((id, j) => {
      ids[i * width + j] = BigInt(id);
      mask[i * width + j] = 1n;
    }));
    const shape = [wrapped.length, width];
    const names = this.session.inputs.map((s) => s.name);
    const feeds: Record<string, unknown> = {};
    if (names.includes("input_ids")) feeds["input_ids"] = new this.Tensor("int64", ids, shape);
    if (names.includes("attention_mask")) feeds["attention_mask"] = new this.Tensor("int64", mask, shape);
    if (names.includes("token_type_ids")) feeds["token_type_ids"] = new this.Tensor("int64", new BigInt64Array(ids.length), shape);
    if (Object.keys(feeds).length === 0) feeds[names[0]!] = new this.Tensor("int64", ids, shape);
    const outputs = (await this.session.run(feeds)) as Record<string, OrtTensor>;
    const heads = new Map<string, OrtTensor>();
    this.session.outputs.forEach((spec, i) => heads.set(this.heads[i]!, outputs[spec.name]!));
    return heads;
  }

  #assemble(piece: string, offsets: [number, number][], heads: Map<string, OrtTensor>, row: number, leads: number[]): string {
    const width = offsets.length;
    const shift = this.#wraps ? 1 : 0;
    const wantsMarks = this.wants.has("punctuation");
    const wantsCase = this.wants.has("capitalisation");
    let post = wantsMarks ? this.#marks("post_punctuation", heads, row, width, shift) : null;
    const pre = wantsMarks ? this.#marks("pre_punctuation", heads, row, width, shift) : null;
    let caps = wantsCase ? flags(heads.get("capitalisation"), row, width, shift, leads) : null;
    if (this.vocabulary.wordLevel) [post, caps] = perWord(offsets, post, caps);
    let corrected = applyLabels(piece, offsets, post, pre, caps);

    if (wantsCase && !this.heads.includes("capitalisation")) corrected = capitaliseSentences(corrected);
    return corrected;
  }

  #marks(role: string, heads: Map<string, OrtTensor>, row: number, width: number, shift: number): string[] | null {
    const tensor = heads.get(role);
    const labels = this.labels[role];
    if (!tensor || !labels || labels.length === 0) return null;
    return argmaxRow(tensor, row, width, shift).map((v) => (v >= 0 && v < labels.length ? labels[v]! : NO_MARK));
  }

  async close(): Promise<void> {
    await this.bundle.close();
  }
}

/** The chosen class per real position of one row: `[B, T, C]` logits, or `[B, T]` indices. */
function argmaxRow(tensor: OrtTensor, row: number, width: number, shift: number): number[] {
  const dims = tensor.dims;
  const data = tensor.data;
  const out: number[] = [];
  if (dims.length >= 3) {
    const [, steps, classes] = [dims[0]!, dims[1]!, dims[dims.length - 1]!];
    for (let t = shift; t < steps && out.length < width; t += 1) {
      const base = (row * steps + t) * classes;
      let best = 0;
      for (let c = 1; c < classes; c += 1) if (Number(data[base + c]) > Number(data[base + best])) best = c;
      out.push(best);
    }
  } else if (dims.length === 2) {
    const steps = dims[1]!;
    for (let t = shift; t < steps && out.length < width; t += 1) out.push(Number(data[row * steps + t]));
  }
  return out;
}

/** The casing head: one flag per subword, or one per character for a boolean `[B, T, C]` head. */
function flags(tensor: OrtTensor | undefined, row: number, width: number, shift: number, leads: number[]): (boolean | boolean[])[] | null {
  if (!tensor) return null;
  if (tensor.dims.length === 3 && tensor.type === "bool") {
    const [, steps, chars] = tensor.dims as [number, number, number];
    const out: boolean[][] = [];
    for (let t = shift; t < steps && out.length < width; t += 1) {
      const i = out.length;
      const from = i < leads.length ? leads[i]! : 0;
      const perChar: boolean[] = [];
      for (let c = from; c < chars; c += 1) perChar.push(Boolean(tensor.data[(row * steps + t) * chars + c]));
      out.push(perChar);
    }
    return out;
  }
  return argmaxRow(tensor, row, width, shift).map((v) => v !== 0);
}

/** Word-level labels: the mark of each word's last subword, the case of its first. */
function perWord(offsets: [number, number][], post: string[] | null, caps: (boolean | boolean[])[] | null): [string[] | null, (boolean | boolean[])[] | null] {
  const count = offsets.length;
  const last = offsets.map((_, i) => i === count - 1 || offsets[i + 1]![0] !== offsets[i]![1]);
  const first = offsets.map((_, i) => i === 0 || last[i - 1]!);
  return [post && post.map((m, i) => (i < count && last[i] ? m : "")), caps && caps.map((f, i) => (i < count && first[i] ? f : false))];
}

/** Task prefixes the T5 checkpoints were fine-tuned with; changing them changes the output. */
export const TASK_PREFIXES: Readonly<Record<string, string>> = {
  "t5-grammar": "grammar: ",
  "t5-gec": "gec: ",
  "t5-synthesis": "",
  "bart-spelling": "",
  "t5-spelling": "",
};

export class RewriteStage implements CorrectionStage {
  readonly kind = "rewrite" as const;
  readonly #engine: GenerationEngine;
  readonly #prefix: string;

  constructor(
    readonly bundle: ModelBundle,
    readonly vocabulary: Vocabulary,
    readonly Tensor: TensorFactory,
    readonly modelName: string,
    family: string,
    readonly maxLength: number,
    readonly config: RewriteConfig,
  ) {
    this.#engine = new GenerationEngine(bundle, Tensor);
    this.#prefix = TASK_PREFIXES[family] ?? "";
  }

  get backend(): string {
    return this.bundle.backend;
  }

  tokenLengths(pieces: string[]): number[] {
    return pieces.map((p) => Math.min(this.vocabulary.encode(this.#prefix + p).length, this.maxLength));
  }

  /**
   * Where decoding begins: the config's `decoder_start_token_id` (BART: `</s>`), else the pad token
   * (T5). Python always takes the pad token, wrong for BART.
   */
  #decoderStart(): number {
    for (const file of ["generation_config.json", "config.json"]) {
      const text = this.bundle.assetText(file);
      if (!text) continue;
      try {
        const value = (JSON.parse(text) as { decoder_start_token_id?: unknown }).decoder_start_token_id;
        if (typeof value === "number") return value;
      } catch {
        // Not fatal: carry on.
      }
    }
    return this.vocabulary.padId;
  }

  /**
   * Each piece rewritten on its own (a padded batch pays for its longest row at every step);
   * returned as decoded, unjudged — the corrector guards it.
   */
  async apply(pieces: string[], signal?: AbortSignal): Promise<string[]> {
    const eos = this.vocabulary.eosId;
    const start = this.#decoderStart();
    const out: string[] = [];
    for (const piece of pieces) {
      let ids = this.vocabulary.encode(this.#prefix + piece).slice(0, this.maxLength - 1);

      if (eos !== null && ids[ids.length - 1] !== eos) ids = [...ids, eos];
      const shape = [1, ids.length];
      const encoded = await this.#engine.encode({
        input_ids: new this.Tensor("int64", BigInt64Array.from(ids, (i) => BigInt(i)), shape),
        attention_mask: new this.Tensor("int64", new BigInt64Array(ids.length).fill(1n), shape),
      });
      const result = await this.#engine.generate({
        prompts: [[start]],
        encoderOutput: encoded,
        strategy: this.config.strategy === "beam" && this.config.beamSize > 1 ? "beam" : "greedy",
        beamSize: this.config.beamSize,
        lengthPenalty: this.config.lengthPenalty,
        repetitionPenalty: this.config.repetitionPenalty,

        maxNewTokens: ids.length + this.config.maxNewTokens,
        temperature: 0,
        eosTokenId: eos,
        padTokenId: this.vocabulary.padId,
        ...(signal ? { signal } : {}),
      });
      out.push(this.vocabulary.decode([...(result.sequences[0] ?? [])]).trim() || piece);
    }
    return out;
  }

  async close(): Promise<void> {
    await this.bundle.close();
  }
}
