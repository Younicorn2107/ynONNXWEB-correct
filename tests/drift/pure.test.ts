import { existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import { buildPlan, describePlan } from "../../src/plan.js";
import { CorrectRegistry } from "../../src/registry.js";
import { Segmenter } from "../../src/segmenter.js";
import { SentencePieceVocabulary, readLabelMap, yamlList } from "../../src/stages.js";
import { applyLabels, applyLengthGuard, capitaliseSentences, correctMarkdown, describeViolation, lengthBatches, restoreSpans, splitFenced, wordSignature } from "../../src/text.js";
import { featuresFromNames, rewriteConfig, type Feature } from "../../src/types.js";

const here = dirname(fileURLToPath(import.meta.url));
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const f = JSON.parse(readFileSync(join(here, "..", "fixtures", "pure.json"), "utf-8")) as any;
const SPM = join(here, "..", ".cache", "opus-mt-fr-en.source.spm");

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const camel = (o: Record<string, any>) => Object.fromEntries(Object.entries(o).map(([k, v]) => [k.replace(/_(\w)/g, (_, c: string) => c.toUpperCase()), v]));

describe("segmentation, against ynonnx_correct", () => {
  it("cuts and splices the same pieces", () => {
    for (const { config, rows } of f.segmentations) {
      const segmenter = new Segmenter(camel(config));
      for (const { text, pieces, spliced } of rows) {
        expect(segmenter.split(text), `${JSON.stringify(config)} ${JSON.stringify(text)}`).toEqual(pieces);
        expect(segmenter.splice(text, segmenter.spans(text), pieces.map((p: string) => p.toUpperCase()))).toBe(spliced);
      }
    }
  });
});

describe("labels and the verbatim contract", () => {
  it("applies labels as apply_labels", () => {
    for (const { text, offsets, post, pre, caps, out } of f.labelled) expect(applyLabels(text, offsets, post, pre, caps)).toBe(out);
  });

  it("capitalises sentences and signs words the same way", () => {
    for (const [text, out] of Object.entries(f.capitalised)) expect(capitaliseSentences(text)).toBe(out);
    for (const [text, out] of Object.entries(f.signatures)) expect(wordSignature(text)).toBe(out);
    expect(describeViolation("the cat sat on the mat", "the dog sat on the mat")).toBe(f.violation);
  });

  it("guards rewrite lengths", () => {
    for (const { config, before, after, kept, reverted } of f.guards) {
      expect(applyLengthGuard(before, after, rewriteConfig(camel(config)))).toEqual([kept, reverted]);
    }
  });
});

describe("Markdown and batching", () => {
  it("keeps code out of the correction", async () => {
    const { body, fenced, corrected } = f.markdown;
    expect(splitFenced(body)).toEqual(fenced);
    const fix = async (texts: string[]) => texts.map((t) => `${capitaliseSentences(t.replace(" inside", ", inside").replace("make release", "MAKE RELEASE"))}.`);
    expect(await correctMarkdown(body, fix)).toBe(corrected);
    expect(restoreSpans("say `hi there` now", "Say `HI there`, now.", [[4, 14]])).toBe(f.restored);
  });

  it("batches by length", () => {
    const lengths = [5, 1, 9, 3, 3, 12, 7, 2];
    expect([...lengthBatches(lengths, 20, { maxBatchItems: 3 })]).toEqual(f.batches.budget);
    expect([...lengthBatches(lengths, 20, { batchSize: 2, maxBatchItems: 3 })]).toEqual(f.batches.fixed);
  });
});

describe("plans", () => {
  it("plans the same passes, or refuses the same way", () => {
    const registry = new CorrectRegistry(undefined, () => true);
    for (const c of f.plans) {
      const run = () => describePlan(buildPlan(featuresFromNames(c.features as Feature[]), registry, { punctuationModel: c.pm, rewriteModel: c.rm, language: c.language, prefer: c.prefer }));
      if (c.error) expect(run, JSON.stringify(c)).toThrow(expect.objectContaining({ name: c.error }));
      else expect(run(), JSON.stringify(c)).toBe(c.plan);
    }
  });

  it("takes xlm-punct-cap-seg for French when punctuate-all cannot be fetched", () => {
    expect(new CorrectRegistry().resolvePunctuation("fr")).toBe("xlm-punct-cap-seg");
  });
});

describe("label maps and vocabularies", () => {
  it("reads id2label and NeMo YAML as Python", () => {
    const json = JSON.stringify({ id2label: { "1": ".", "0": "0", "2": ",", "10": "?", "3": "O" } });
    expect(readLabelMap({ configJson: json }, ["post_punctuation"])).toEqual(f.labelMaps.json);
    expect(readLabelMap({ configYaml: f.labelMaps.yamlText }, ["pre_punctuation", "post_punctuation", "capitalisation", "segmentation"])).toEqual(f.labelMaps.yaml);
    expect(yamlList("other: [1, 2]\n", "other")).toEqual(["1", "2"]);

    const config = ['post_labels: [', '  "<NULL>",', '  ".",', '  ",",', "  '''',", '  "\\"",', "]", "next: 1"].join("\n");
    expect(yamlList(config, "post_labels")).toEqual(["<NULL>", ".", ",", "'", '"']);
  });

  it.skipIf(!existsSync(SPM))("recovers SentencePiece spans as Python", () => {
    for (const lowercase of [false, true]) {
      const vocab = new SentencePieceVocabulary(new Uint8Array(readFileSync(SPM)), lowercase);
      for (const row of f.spOffsets.filter((r: { lowercase: boolean }) => r.lowercase === lowercase)) {
        const [ids, offsets] = vocab.encodeWithOffsets(row.text);
        expect(ids, row.text).toEqual(row.ids);

        const toPoints = (i: number) => Array.from(row.text.slice(0, i)).length;
        expect(offsets.map(([s, e]) => [toPoints(s), toPoints(e)]), row.text).toEqual(row.offsets);
        expect(vocab.leads(row.text)).toEqual(row.leads);
      }
    }
  });
});
