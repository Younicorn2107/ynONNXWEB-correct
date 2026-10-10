/**
 * The pure parts of the pipeline: labels spliced into the caller's string, the verbatim contract,
 * capitals from marks, the two output guards, Markdown, and length batching. Ports of
 * `ynonnx_correct.core.{labels, verbatim, casing, guards, markdown, batcher}`.
 */

import { splitLinesKeepEnds, type Logger } from "@youni/onnxweb-core/core";

import { TERMINALS, TRAILING, isAlnum, isAlpha, isSpace } from "./segmenter.js";
import { RewriteLengthError, VerbatimViolationError, withinBounds, type RewriteConfig } from "./types.js";

export const NO_MARK = "";
/** The punct_cap_seg label that puts a full stop after every letter of the subword. */
export const ACRONYM = "<ACRONYM>";
/** Marks a post label may sit next to without doubling. */
const EXISTING_MARKS = ".,;:!?…。，、；：！？";

const isUpper = (c: string) => c !== c.toLowerCase() && c === c.toUpperCase();

/**
 * Marks inserted and case applied at the positions `offsets` names (UTF-16 spans of `text`).
 * Everything between subwords is copied through, each subword is copied from the original: the only
 * additions are marks and case.
 */
export function applyLabels(
  text: string,
  offsets: readonly [number, number][],
  post: readonly string[] | null = null,
  pre: readonly string[] | null = null,
  caps: readonly (boolean | readonly boolean[])[] | null = null,
): string {
  if (offsets.length === 0) return text;
  const out: string[] = [];
  let cursor = 0;
  offsets.forEach(([start, end], index) => {
    if (start < cursor || end < start || end > text.length) return;
    out.push(text.slice(cursor, start));
    const prefix = mark(pre, index);
    if (prefix) out.push(prefix);
    let token = text.slice(start, end);
    const flags = caps !== null && index < caps.length ? caps[index]! : false;
    if (Array.isArray(flags)) token = Array.from(token).map((c, i) => (i < flags.length && flags[i] ? c.toUpperCase() : c)).join("");
    else if (flags) token = capitalise(token);
    if (post !== null && index < post.length && post[index] === ACRONYM) {
      out.push(Array.from(token).map((c) => (isAlnum(c) ? `${c}.` : c)).join(""));
      cursor = end;
      return;
    }
    out.push(token);
    const suffix = mark(post, index);
    if (suffix && !alreadyMarked(text, start, end)) out.push(suffix);
    cursor = end;
  });
  out.push(text.slice(cursor));
  return out.join("");
}

function mark(labels: readonly string[] | null, index: number): string {
  if (labels === null || index >= labels.length) return NO_MARK;
  const label = labels[index] || NO_MARK;
  if (label && Array.from(label).some(isAlnum)) {
    throw new VerbatimViolationError(
      `the tagger produced the label '${label}', which contains a letter or a digit. A punctuation head's labels are marks; one carrying text means the label map does not belong to this head, and applying it would insert words into the caller's transcript.`,
    );
  }
  return label;
}

function alreadyMarked(text: string, start: number, end: number): boolean {
  if (end > start && EXISTING_MARKS.includes(text[end - 1]!)) return true;
  let index = end;
  while (index < text.length && isSpace(text[index]!)) index += 1;
  return index < text.length && EXISTING_MARKS.includes(text[index]!);
}

/** The first LETTER uppercased, the rest left alone. */
function capitalise(token: string): string {
  const chars = Array.from(token);
  for (const [i, c] of chars.entries()) {
    if (!isAlpha(c)) continue;
    if (isUpper(c)) return token;
    chars[i] = c.toUpperCase();
    return chars.join("");
  }
  return token;
}

/** Lowercase alphanumerics only: what a punctuation pass may not touch. */
export function wordSignature(text: string): string {
  return Array.from(text)
    .filter(isAlnum)
    .map((c) => c.toLowerCase())
    .join("");
}

export function isVerbatim(original: string, corrected: string): boolean {
  return wordSignature(original) === wordSignature(corrected);
}

/** Index in the signature where the two texts first disagree, or -1. */
export function firstDivergence(original: string, corrected: string): number {
  const left = Array.from(wordSignature(original));
  const right = Array.from(wordSignature(corrected));
  for (let i = 0; i < Math.min(left.length, right.length); i += 1) if (left[i] !== right[i]) return i;
  return left.join("") === right.join("") ? -1 : Math.min(left.length, right.length);
}

export function describeViolation(original: string, corrected: string, window = 24): string {
  const index = firstDivergence(original, corrected);
  if (index < 0) return "no divergence";
  const left = Array.from(wordSignature(original));
  const right = Array.from(wordSignature(corrected));
  const start = Math.max(0, index - window);
  return `at word-signature offset ${index}: expected ...${left.slice(start, index + window).join("")}... but got ...${right.slice(start, index + window).join("")}...`;
}

/** The first letter of the text and of each sentence uppercased, when lowercase. */
export function capitaliseSentences(text: string): string {
  if (!text) return text;
  const chars = Array.from(text);
  let pending = true;
  for (const [i, c] of chars.entries()) {
    if (pending && isAlpha(c)) {
      chars[i] = c.toUpperCase();
      pending = false;
      continue;
    }
    if (TERMINALS.includes(c)) pending = true;
    else if (pending && isSpace(c)) continue;
    else if (pending && TRAILING.includes(c)) continue;
    else if (isAlnum(c)) pending = false;
  }
  return chars.join("");
}

/** Every piece of a non-destructive stage checked word for word. */
export function enforceVerbatim(before: readonly string[], after: readonly string[], model: string, kind: string): void {
  before.forEach((original, i) => {
    const corrected = after[i]!;
    if (isVerbatim(original, corrected)) return;
    throw new VerbatimViolationError(
      `'${model}' runs as a ${kind} stage, which may only add marks and change case, and it changed a word — ${describeViolation(original, corrected)}. Either the label map does not belong to this graph's heads, or the catalogue declares the wrong kind for this model.`,
    );
  });
}

/** Each rewrite accepted, or sent back to its original per the policy: `[kept, reverted]`. */
export function applyLengthGuard(before: readonly string[], after: readonly string[], config: RewriteConfig, logger?: Logger): [string[], number] {
  const kept: string[] = [];
  let reverted = 0;
  before.forEach((original, i) => {
    const cleaned = after[i]!.trim();
    const originalChars = Array.from(original).length;
    const cleanedChars = Array.from(cleaned).length;
    if (cleaned && withinBounds(config, originalChars, cleanedChars)) {
      kept.push(cleaned);
      return;
    }
    const detail = `a rewrite of ${cleanedChars} characters came back for an input of ${originalChars}, outside the [${config.minLengthRatio}, ${config.maxLengthRatio}] guard`;
    if (config.onLengthViolation === "error") throw new RewriteLengthError(`${detail}: '${Array.from(original).slice(0, 80).join("")}'`);
    if (config.onLengthViolation === "accept") {
      logger?.warn(`${detail}; accepted as configured`);
      kept.push(cleaned);
      return;
    }
    reverted += 1;
    logger?.debug?.(`${detail}; kept the original piece`);
    kept.push(original);
  });
  return [kept, reverted];
}

const FENCE = /^\s*(```|~~~)/u;
const INLINE_CODE = /`[^`\n]+`/gu;

/** The body as consecutive `[text, isCode]` chunks, fences included in the code. */
export function splitFenced(body: string): [string, boolean][] {
  const chunks: [string, boolean][] = [];
  let current: string[] = [];
  let inCode = false;
  for (const line of splitLinesKeepEnds(body)) {
    if (FENCE.test(line)) {
      if (!inCode) {
        if (current.length > 0) chunks.push([current.join(""), false]);
        current = [line];
        inCode = true;
      } else {
        current.push(line);
        chunks.push([current.join(""), true]);
        current = [];
        inCode = false;
      }
      continue;
    }
    current.push(line);
  }
  if (current.length > 0) chunks.push([current.join(""), inCode]);
  return chunks;
}

/**
 * `corrected` with each `[start, end)` span (code points) of `original` put back as it was,
 * aligning the two case-insensitively; returned as it is when a word changed.
 */
export function restoreSpans(original: string, corrected: string, spans: readonly [number, number][]): string {
  if (spans.length === 0) return corrected;
  const source = Array.from(original);
  const target = Array.from(corrected);
  const where: number[] = [];
  let j = 0;
  for (const c of source) {
    while (j < target.length && target[j]!.toLowerCase() !== c.toLowerCase()) {
      if (isAlnum(target[j]!)) return corrected;
      j += 1;
    }
    if (j === target.length) return corrected;
    where.push(j);
    j += 1;
  }
  let out = target;
  for (const [start, end] of [...spans].sort((a, b) => b[0] - a[0] || b[1] - a[1])) {
    if (end <= start) continue;
    out = [...out.slice(0, where[start]!), ...source.slice(start, end), ...out.slice(where[end - 1]! + 1)];
  }
  return out.join("");
}

/** The body with its prose corrected in one pool, its code untouched. */
export async function correctMarkdown(body: string, fix: (texts: string[]) => Promise<string[]>): Promise<string> {
  const chunks = splitFenced(body);
  const prose = chunks.filter(([text, code]) => !code && text.trim()).map(([text]) => text);
  const fixed = prose.length > 0 ? await fix(prose) : [];
  let next = 0;
  return chunks
    .map(([text, code]) => {
      if (code || !text.trim()) return text;
      const corrected = fixed[next++]!;
      const spans = [...text.matchAll(INLINE_CODE)].map((m): [number, number] => {
        const start = Array.from(text.slice(0, m.index)).length;
        return [start, start + Array.from(m[0]).length];
      });
      return keepEdges(text, restoreSpans(text, corrected, spans));
    })
    .join("");
}

function keepEdges(original: string, corrected: string): string {
  const head = original.slice(0, original.length - original.trimStart().length);
  const tail = original.slice(original.trimEnd().length);
  return head + corrected.trim() + tail;
}

/**
 * Indices grouped by similar length, under a token budget (rows × longest row) and an item ceiling.
 */
export function* lengthBatches(lengths: readonly number[], tokenBudget: number, options: { batchSize?: number | null; maxBatchItems?: number } = {}): Generator<number[]> {
  if (lengths.length === 0) return;
  const maxItems = Math.max(Math.trunc(options.maxBatchItems ?? 128), 1);
  const order = lengths.map((_, i) => i).sort((a, b) => lengths[a]! - lengths[b]! || a - b);
  let current: number[] = [];
  let longest = 0;
  for (const position of order) {
    let candidate = Math.max(longest, lengths[position]!);
    const full =
      options.batchSize !== null && options.batchSize !== undefined
        ? current.length >= options.batchSize
        : current.length > 0 && ((current.length + 1) * candidate > tokenBudget || current.length >= maxItems);
    if (full) {
      yield current;
      current = [];
      candidate = lengths[position]!;
    }
    current.push(position);
    longest = candidate;
  }
  if (current.length > 0) yield current;
}
