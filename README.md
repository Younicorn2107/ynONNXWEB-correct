# onnxweb-correct

Text correction in the browser: punctuation, capitals, sentence boundaries, grammar.

A pipeline of stages: a token-classification model restores punctuation and capitals (47 languages), a seq2seq model rewrites grammar, and every stage is checked to keep the words it must not change. Subtitles and Markdown keep their structure.

```ts
import * as ort from "onnxruntime-web";
import { TextCorrector } from "@youni/onnxweb-correct";

const corrector = new TextCorrector({ ort, Tensor: ort.Tensor, language: "fr" });
const result = await corrector.correct("bonjour je m'appelle loïc et toi");
console.log(result.text);
```

## Install

Builds against `@youni/onnxweb-core`:

```sh
npm install
npm run build   # dist/
npm test        # unit tests (Node, onnxruntime-web on WebAssembly)
```

## Licence

Apache-2.0 for the code. Model weights keep their own licence (listed above where it matters).
