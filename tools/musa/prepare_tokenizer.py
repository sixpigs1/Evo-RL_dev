"""Export the public original PaliGemma SentencePiece tokenizer for offline Pi0.5."""

import hashlib
import json
import urllib.request
from pathlib import Path

from transformers import GemmaTokenizer

root = Path("/workspace/storage/models/paligemma-tokenizer")
root.mkdir(parents=True, exist_ok=True)
url = "https://storage.googleapis.com/big_vision/paligemma_tokenizer.model"
model = root / "tokenizer.model"
if not model.exists():
    urllib.request.urlretrieve(url, model)
tokenizer = GemmaTokenizer(vocab_file=str(model), add_bos_token=True, add_eos_token=False)
assert tokenizer.vocab_size == 257152
assert (tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id) == (2, 1, 0)
tokenizer.save_pretrained(root)
(root / "source.json").write_text(
    json.dumps(
        {
            "url": url,
            "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
            "vocab_size": tokenizer.vocab_size,
            "note": "Original public OpenPI/PaliGemma tokenizer, no substituted vocabulary",
        },
        indent=2,
    )
)
