#!/usr/bin/env bash
set -euo pipefail

status=0

if grep -R -nEi   '^[[:space:]]*(llama[-_]?cpp|llama_cpp|onnxruntime|ort|tensorflow|tflite|candle([_-][a-z0-9_-]+)?|burn)[[:space:]]*='   crates --include='Cargo.toml'
then
  echo "Forbidden third-party AI/model runtime dependency detected in Rust manifests." >&2
  status=1
fi

if grep -R -nEi   '(api\.openai\.com|api\.anthropic\.com|generativelanguage\.googleapis\.com|api\.cohere\.(ai|com))'   crates platform   --include='*.rs' --include='*.java' --include='*.kt' --include='*.gradle'
then
  echo "Hosted AI API endpoint detected in canonical runtime source." >&2
  status=1
fi

exit "${status}"
