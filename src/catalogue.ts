/** The text-correction catalogue. The port of `ynonnx_correct.constants.model_registry`. */

import type { Feature, StageKind, TokenizerKind } from "./types.js";

export const PUNCT_AND_CASE: readonly Feature[] = ["punctuation", "capitalisation"];
export const SPELLING_AND_GRAMMAR: readonly Feature[] = ["spelling", "grammar"];

export const PUNCT_CAP_SEG_47_LANGUAGES: readonly string[] = [
  "af", "am", "ar", "bg", "bn", "de", "el", "en", "es", "et", "fa", "fi",
  "fr", "gu", "hi", "hr", "hu", "id", "is", "it", "ja", "kk", "kn", "ko",
  "ky", "lt", "lv", "mk", "ml", "mr", "nl", "or", "pa", "pl", "ps", "pt",
  "ro", "ru", "rw", "so", "sr", "sw", "ta", "te", "tr", "uk", "zh",
];

export interface CorrectionEntry {
  hfId: string;
  /**
   * The published ONNX repository; null when there is none (a server resolver can serve an export).
   */
  onnxRepo: string | null;
  onnxSubfolder: string | null;
  /** Exact graph filename in the repository. */
  graphFile: string;
  /** Exact vocabulary filename, and which reader opens it. */
  tokenizerFile: string;
  tokenizerKind: TokenizerKind;
  family: string;
  kind: StageKind;
  features: readonly Feature[];
  /** Subword positions the encoder accepts. */
  maxLength: number;
  /** For a tagging graph: what each output means, in graph order. */
  heads: readonly string[] | null;
  languages: readonly string[];
  paramsM: number;
  licence: string;
  notes: string | null;
}

export const MODEL_CATALOGUE: Readonly<Record<string, CorrectionEntry>> = {
  "xlm-punct-cap-seg": {
    hfId: "1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase",
    onnxRepo: "1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase",
    onnxSubfolder: null,
    graphFile: "model.onnx",
    tokenizerFile: "sp.model",
    tokenizerKind: "sentencepiece",
    family: "punct-cap-seg",
    kind: "tagging",
    features: PUNCT_AND_CASE,
    maxLength: 512,
    heads: ["pre_punctuation", "post_punctuation", "capitalisation", "segmentation"],
    languages: PUNCT_CAP_SEG_47_LANGUAGES,
    paramsM: 278,
    licence: "apache-2.0",
    notes: "47 languages, the quality reference. Pre-punctuation puts the opening ¿ and ¡ in Spanish.",
  },
  "punct-cap-seg-47": {
    hfId: "1-800-BAD-CODE/punct_cap_seg_47_language",
    onnxRepo: "1-800-BAD-CODE/punct_cap_seg_47_language",
    onnxSubfolder: null,
    graphFile: "punct_cap_seg_47lang.onnx",
    tokenizerFile: "spe_unigram_64k_lowercase_47lang.model",
    tokenizerKind: "sentencepiece",
    family: "punct-cap-seg",
    kind: "tagging",
    features: PUNCT_AND_CASE,
    maxLength: 256,
    heads: ["pre_punctuation", "post_punctuation", "capitalisation", "segmentation"],
    languages: PUNCT_CAP_SEG_47_LANGUAGES,
    paramsM: 100,
    licence: "apache-2.0",
    notes: "The predecessor of xlm-punct-cap-seg, a third of the weight; fed lowercased text.",
  },
  "punct-cap-seg-en": {
    hfId: "1-800-BAD-CODE/punctuation_fullstop_truecase_english",
    onnxRepo: "1-800-BAD-CODE/punctuation_fullstop_truecase_english",
    onnxSubfolder: null,
    graphFile: "punct_cap_seg_en.onnx",
    tokenizerFile: "spe_32k_lc_en.model",
    tokenizerKind: "sentencepiece",
    family: "punct-cap-seg",
    kind: "tagging",
    features: PUNCT_AND_CASE,
    maxLength: 256,

    heads: ["pre_punctuation", "post_punctuation", "capitalisation", "segmentation"],
    languages: ["en"],
    paramsM: 22,
    licence: "apache-2.0",
    notes: "English only. Its pre-punctuation head is exported but untrained (\"not actually used\", says its config).",
  },
  "punct-cap-seg-romance": {
    hfId: "1-800-BAD-CODE/punctuation_fullstop_truecase_romance",
    onnxRepo: "1-800-BAD-CODE/punctuation_fullstop_truecase_romance",
    onnxSubfolder: null,
    graphFile: "model.onnx",
    tokenizerFile: "sp.model",
    tokenizerKind: "sentencepiece",
    family: "punct-cap-seg",
    kind: "tagging",
    features: PUNCT_AND_CASE,
    maxLength: 256,
    heads: ["pre_punctuation", "post_punctuation", "capitalisation", "segmentation"],
    languages: ["es", "ca", "fr", "pt", "it", "ro"],
    paramsM: 100,
    licence: "apache-2.0",
    notes: "Six Romance languages; worth it for Spanish and its inverted marks.",
  },
  "fullstop-large": {
    hfId: "oliverguhr/fullstop-punctuation-multilang-large",
    onnxRepo: "oliverguhr/fullstop-punctuation-multilang-large",
    onnxSubfolder: "onnx",
    graphFile: "model.onnx",
    tokenizerFile: "tokenizer.json",
    tokenizerKind: "tokenizers",
    family: "fullstop",
    kind: "tagging",
    features: PUNCT_AND_CASE,
    maxLength: 512,
    heads: ["post_punctuation"],
    languages: ["en", "de", "fr", "it"],
    paramsM: 560,
    licence: "mit",
    notes: "XLM-R large, 2.2 GB of float32 weights in a model.onnx_data sidecar. Punctuation only — case comes from the rule pass.",
  },
  "punctuate-all": {
    hfId: "kredor/punctuate-all",
    onnxRepo: null,
    onnxSubfolder: null,
    graphFile: "model.onnx",
    tokenizerFile: "tokenizer.json",
    tokenizerKind: "tokenizers",
    family: "fullstop",
    kind: "tagging",
    features: PUNCT_AND_CASE,
    maxLength: 512,
    heads: ["post_punctuation"],
    languages: ["en", "de", "fr", "es", "bg", "it", "pl", "nl", "cs", "pt", "sk", "sl"],
    paramsM: 278,
    licence: "mit",
    notes: "No published ONNX: served only by a resolver that holds an export.",
  },
  "sbd-49": {
    hfId: "1-800-BAD-CODE/sentence_boundary_detection_multilang",
    onnxRepo: "1-800-BAD-CODE/sentence_boundary_detection_multilang",
    onnxSubfolder: null,
    graphFile: "sbd_49lang_bert_small.onnx",
    tokenizerFile: "spe_mixed_case_64k_49lang.model",
    tokenizerKind: "sentencepiece",
    family: "punct-cap-seg",
    kind: "tagging",
    features: [],
    maxLength: 256,
    heads: ["segmentation"],
    languages: [
      "af", "ar", "bg", "bn", "bs", "ca", "cs", "da", "de", "el", "en", "eo",
      "es", "et", "eu", "fi", "fr", "gl", "he", "hi", "hr", "hu", "id", "is",
      "it", "ja", "ka", "ko", "lt", "lv", "mk", "ml", "ms", "nl", "no", "pl",
      "pt", "ro", "ru", "si", "sk", "sl", "sq", "sv", "th", "tr", "uk", "vi",
      "zh",
    ],
    paramsM: 20,
    licence: "apache-2.0",
    notes: "Boundaries only, 49 languages. Serves no feature; it feeds the segmenter.",
  },
  "grammar-t5-base": {
    hfId: "vennify/t5-base-grammar-correction",
    onnxRepo: "onnx-community/t5-base-grammar-correction-ONNX",
    onnxSubfolder: "onnx",
    graphFile: "encoder_model.onnx",
    tokenizerFile: "tokenizer.json",
    tokenizerKind: "tokenizers",
    family: "t5-grammar",
    kind: "rewrite",
    features: SPELLING_AND_GRAMMAR,
    maxLength: 512,
    heads: null,
    languages: ["en"],
    paramsM: 223,
    licence: "cc-by-nc-sa-4.0",
    notes: "Not chosen by auto (non-commercial licence); grammar-synthesis-small is.",
  },
  "grammar-synthesis-small": {
    hfId: "pszemraj/grammar-synthesis-small",
    onnxRepo: "onnx-community/grammar-synthesis-small-ONNX",
    onnxSubfolder: "onnx",
    graphFile: "encoder_model.onnx",
    tokenizerFile: "tokenizer.json",
    tokenizerKind: "tokenizers",
    family: "t5-synthesis",
    kind: "rewrite",
    features: SPELLING_AND_GRAMMAR,
    maxLength: 512,
    heads: null,
    languages: ["en"],
    paramsM: 77,
    licence: "apache-2.0",
    notes: "Trained to leave a clean sentence alone.",
  },
  "gector-t5": {
    hfId: "prithivida/grammar_error_correcter_v1",
    onnxRepo: "onnx-community/grammar_error_correcter_v1-ONNX",
    onnxSubfolder: "onnx",
    graphFile: "encoder_model.onnx",
    tokenizerFile: "tokenizer.json",
    tokenizerKind: "tokenizers",
    family: "t5-gec",
    kind: "rewrite",
    features: SPELLING_AND_GRAMMAR,
    maxLength: 512,
    heads: null,
    languages: [],
    paramsM: 223,
    licence: "apache-2.0",
    notes: "The Gramformer checkpoint; no language confirmed, so never chosen by auto.",
  },
  "grammar-flan-t5-large": {
    hfId: "pszemraj/flan-t5-large-grammar-synthesis",
    onnxRepo: null,
    onnxSubfolder: null,
    graphFile: "encoder_model.onnx",
    tokenizerFile: "tokenizer.json",
    tokenizerKind: "tokenizers",
    family: "t5-synthesis",
    kind: "rewrite",
    features: SPELLING_AND_GRAMMAR,
    maxLength: 512,
    heads: null,
    languages: ["en"],
    paramsM: 783,
    licence: "cc-by-nc-sa-4.0",
    notes: "Best English quality, ten times the weight; its published onnx/ is one monolithic graph. Non-commercial.",
  },
  "spelling-en": {
    hfId: "oliverguhr/spelling-correction-english-base",
    onnxRepo: "oliverguhr/spelling-correction-english-base",
    onnxSubfolder: "onnx",
    graphFile: "encoder_model.onnx",
    tokenizerFile: "tokenizer.json",
    tokenizerKind: "tokenizers",
    family: "bart-spelling",
    kind: "rewrite",
    features: ["spelling"],
    maxLength: 1024,
    heads: null,
    languages: ["en"],
    paramsM: 139,
    licence: "mit",
    notes: "Spelling ONLY: asking for grammar too plans a second stage.",
  },
  "spelling-de": {
    hfId: "oliverguhr/spelling-correction-german-base",
    onnxRepo: null,
    onnxSubfolder: null,
    graphFile: "encoder_model.onnx",
    tokenizerFile: "tokenizer.json",
    tokenizerKind: "tokenizers",
    family: "t5-spelling",
    kind: "rewrite",
    features: ["spelling"],
    maxLength: 512,
    heads: null,
    languages: ["de"],
    paramsM: 223,
    licence: "apache-2.0",
    notes: "German. No published ONNX.",
  },
};

/** Convenience names, plus the `org/repo` spellings copied out of a model page. */
export const MODEL_ALIASES: Readonly<Record<string, string>> = {
  punct: "fullstop-large",
  punctuation: "fullstop-large",
  fullstop: "fullstop-large",
  "punct-cap-seg": "xlm-punct-cap-seg",
  truecase: "xlm-punct-cap-seg",
  grammar: "grammar-synthesis-small",
  gramformer: "gector-t5",
  spelling: "spelling-en",
  ...Object.fromEntries(Object.entries(MODEL_CATALOGUE).map(([key, entry]) => [entry.hfId, key])),
  ...Object.fromEntries(Object.entries(MODEL_CATALOGUE).filter(([, e]) => e.onnxRepo).map(([key, entry]) => [entry.onnxRepo!, key])),
};

export const HF_ID_TO_ALIAS: Readonly<Record<string, string>> = Object.fromEntries(Object.entries(MODEL_CATALOGUE).map(([key, entry]) => [entry.hfId, key]));

/** Decoder LLMs prompted to "correct this text": out of scope by design. */
export const LLM_DERIVED_MARKERS: readonly string[] = ["qwen", "mistral", "llama", "gemma", "phi", "gpt", "instruct"];

export interface Preset {
  features: readonly Feature[];
  prefer?: "fast" | "balanced" | "quality";
  computeType?: string;
}

/** Named sets of switches and settings — never a model, so the language still decides. */
export const PRESETS: Readonly<Record<string, Preset>> = {
  transcript: { features: ["punctuation", "capitalisation"] },
  prose: { features: ["punctuation", "capitalisation", "spelling", "grammar"] },
  "grammar-only": { features: ["grammar", "spelling"] },
  cpu: { features: ["punctuation", "capitalisation"], prefer: "fast", computeType: "int8" },
};

/** Subword positions per batch when the batch size is automatic. */
export const DEFAULT_TOKEN_BUDGET = 16384;
export const DEFAULT_MAX_BATCH_ITEMS = 128;
