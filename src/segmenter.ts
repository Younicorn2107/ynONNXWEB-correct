/**
 * Cutting a text into the pieces a model corrects, and putting them back. The port of
 * `ynonnx_correct.core.segmenter`.
 */

import { DEFAULT_SEGMENTATION, type SegmentationConfig } from "./types.js";

/** Marks that can end a sentence; a run of them is one boundary. */
export const TERMINALS = ".!?…。！？";
/** Marks that may trail a terminal and still belong to its sentence. */
export const TRAILING = "\"'”’)]}»";
/** Marks that make a cut inside a window less bad than an arbitrary one. */
export const SOFT_BREAKS = ",;:，、；：";
/** Words whose full stop is part of the word. */
export const ABBREVIATIONS: ReadonlySet<string> = new Set([
  "mr", "mrs", "ms", "dr", "prof", "st", "jr", "sr",
  "m", "mme", "mlle", "me", "pr",
  "hr", "fr", "dipl",
  "etc", "vs", "cf", "al", "ca", "ibid", "op", "cit", "viz",
  "e.g", "i.e", "eg", "ie",
  "no", "nos", "fig", "vol", "ed", "pp", "approx", "dept", "univ",
  "inc", "ltd", "co", "corp",
]);

// eslint-disable-next-line no-control-regex -- Python counts the file/group/record/unit separators as whitespace
export const isSpace = (c: string) => c !== "﻿" && /^[\s\x1c-\x1f\x85]$/u.test(c);
export const isAlpha = (c: string) => /^\p{L}$/u.test(c);
export const isAlnum = (c: string) => /^[\p{L}\p{N}]$/u.test(c);
export const isDigit = (c: string) => /^[\p{Nd}²³¹⁰⁴-⁹₀-₉]$/u.test(c);

export type Span = [number, number];

export class Segmenter {
  readonly config: SegmentationConfig;

  constructor(config: Partial<SegmentationConfig> = {}) {
    this.config = { ...DEFAULT_SEGMENTATION, ...config };
  }

  /** Disjoint `[start, end)` spans (code points) covering every non-blank stretch. */
  spans(text: string): Span[] {
    const chars = Array.from(text);
    if (chars.length === 0 || !text.trim()) return [];
    const pieces = this.config.splitOnSentences && chars.some((c) => TERMINALS.includes(c)) ? this.#sentenceSpans(chars) : [trim(chars, 0, chars.length)];
    const windowed: Span[] = [];
    for (const [start, end] of pieces) if (end > start) windowed.push(...this.#window(chars, start, end));
    return this.#mergeShort(chars, windowed);
  }

  /** The pieces as strings. */
  split(text: string): string[] {
    const chars = Array.from(text);
    return this.spans(text).map(([s, e]) => chars.slice(s, e).join(""));
  }

  #sentenceSpans(chars: string[]): Span[] {
    const spans: Span[] = [];
    let start = 0;
    let index = 0;
    const length = chars.length;
    while (index < length) {
      if (!TERMINALS.includes(chars[index]!)) {
        index += 1;
        continue;
      }
      let end = index;
      while (end + 1 < length && TERMINALS.includes(chars[end + 1]!)) end += 1;
      while (end + 1 < length && TRAILING.includes(chars[end + 1]!)) end += 1;
      if (!isBoundary(chars, index, end)) {
        index = end + 1;
        continue;
      }
      spans.push(trim(chars, start, end + 1));
      start = end + 1;
      index = end + 1;
    }
    if (start < length) spans.push(trim(chars, start, length));
    return spans.filter(([s, e]) => e > s);
  }

  /** One over-long span cut into word windows, on the best break in the window's tail. */
  #window(chars: string[], start: number, end: number): Span[] {
    const words = wordsOf(chars, start, end);
    const limit = this.config.maxWords;
    if (words.length <= limit) return [[start, end]];
    const spans: Span[] = [];
    let index = 0;
    while (index < words.length) {
      const ceiling = Math.min(index + limit, words.length);
      let cut = ceiling;
      if (ceiling < words.length) {
        const lookback = Math.max(1, Math.trunc(limit * this.config.cutSearchRatio));
        cut = bestCut(chars, words, index + 1, ceiling, lookback);
      }
      spans.push([words[index]![0], words[cut - 1]![1]]);
      index = cut;
    }
    return spans;
  }

  /** A piece shorter than `minChars` folded into its neighbour. */
  #mergeShort(chars: string[], spans: Span[]): Span[] {
    const floor = this.config.minChars;
    if (floor <= 0 || spans.length === 0) return spans;
    const size = ([s, e]: Span) => Array.from(chars.slice(s, e).join("").trim()).length;
    const merged: Span[] = [];
    for (const span of spans) {
      if (size(span) >= floor) merged.push(span);
      else if (merged.length > 0) merged[merged.length - 1] = [merged[merged.length - 1]![0], span[1]];
      else merged.push(span);
    }
    if (merged.length > 1 && size(merged[0]!) < floor) {
      merged[1] = [merged[0]![0], merged[1]![1]];
      merged.shift();
    }
    return merged;
  }

  /**
   * Each span's correction substituted into the original; what lies between spans is copied
   * through.
   */
  splice(text: string, spans: readonly Span[], replacements: readonly string[]): string {
    if (spans.length !== replacements.length) throw new Error(`${spans.length} spans for ${replacements.length} replacements`);
    const chars = Array.from(text);
    const out: string[] = [];
    let cursor = 0;
    spans.forEach(([start, end], i) => {
      if (start < cursor) throw new Error(`span (${start}, ${end}) overlaps the previous one, which ended at ${cursor}; splicing would duplicate text`);
      out.push(chars.slice(cursor, start).join(""), replacements[i]!);
      cursor = end;
    });
    out.push(chars.slice(cursor).join(""));
    return out.join("");
  }
}

/**
 * Whether the terminal at `mark` ends a sentence: not a decimal point, an abbreviation, or glued to
 * the next word.
 */
function isBoundary(chars: string[], mark: number, end: number): boolean {
  if (chars[mark] !== "." || end !== mark) return end + 1 >= chars.length || isSpace(chars[end + 1]!);
  const before = mark > 0 ? chars[mark - 1]! : "";
  const after = mark + 1 < chars.length ? chars[mark + 1]! : "";
  if (before && after && isDigit(before) && isDigit(after)) return false;
  if (end + 1 < chars.length && !isSpace(chars[end + 1]!)) return false;
  let wordStart = mark;
  while (wordStart > 0 && !isSpace(chars[wordStart - 1]!)) wordStart -= 1;
  const word = chars.slice(wordStart, mark).join("").trim().toLowerCase();
  return !ABBREVIATIONS.has(word);
}

/** Index to cut before, searching back from `ceiling`: a line break first, then a soft break. */
function bestCut(chars: string[], words: Span[], floor: number, ceiling: number, lookback: number): number {
  const lower = Math.max(floor, ceiling - lookback);
  for (let c = ceiling; c >= lower; c -= 1) if (chars.slice(words[c - 1]![1], words[c]![0]).includes("\n")) return c;
  for (let c = ceiling; c >= lower; c -= 1) if (SOFT_BREAKS.includes(chars[words[c - 1]![1] - 1]!)) return c;
  return ceiling;
}

function wordsOf(chars: string[], start: number, end: number): Span[] {
  const words: Span[] = [];
  let index = start;
  while (index < end) {
    while (index < end && isSpace(chars[index]!)) index += 1;
    if (index >= end) break;
    const wordStart = index;
    while (index < end && !isSpace(chars[index]!)) index += 1;
    words.push([wordStart, index]);
  }
  return words;
}

function trim(chars: string[], start: number, end: number): Span {
  while (start < end && isSpace(chars[start]!)) start += 1;
  while (end > start && isSpace(chars[end - 1]!)) end -= 1;
  return [start, end];
}

/** Whitespace-delimited words. */
export function wordCount(text: string): number {
  // eslint-disable-next-line no-control-regex -- Python counts the file/group/record/unit separators as whitespace
  return text.split(/[\s\x1c-\x1f\x85]+/u).filter(Boolean).length;
}
