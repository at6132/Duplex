import string


SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<UPDATE>", "<SEP>"]

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UPDATE_ID = 3
SEP_ID = 4


class CharTokenizer:
    """Character-level tokenizer with special tokens for duplex LM experiments."""

    def __init__(self):
        self.special_tokens = list(SPECIAL_TOKENS)
        self.chars = list(string.printable)  # 100 printable ASCII characters

        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}

        for i, tok in enumerate(self.special_tokens):
            self.token_to_id[tok] = i
            self.id_to_token[i] = tok

        offset = len(self.special_tokens)
        for i, ch in enumerate(self.chars):
            self.token_to_id[ch] = i + offset
            self.id_to_token[i + offset] = ch

        self.vocab_size = len(self.token_to_id)
        self.pad_id = PAD_ID
        self.bos_id = BOS_ID
        self.eos_id = EOS_ID
        self.update_id = UPDATE_ID
        self.sep_id = SEP_ID

    def encode(self, text: str) -> list[int]:
        ids = []
        i = 0
        while i < len(text):
            matched = False
            for special in self.special_tokens:
                if text[i:i + len(special)] == special:
                    ids.append(self.token_to_id[special])
                    i += len(special)
                    matched = True
                    break
            if not matched:
                ch = text[i]
                if ch in self.token_to_id:
                    ids.append(self.token_to_id[ch])
                i += 1
        return ids

    def decode(self, ids: list[int]) -> str:
        tokens = []
        for token_id in ids:
            if token_id == self.pad_id:
                continue
            tok = self.id_to_token.get(token_id, "")
            tokens.append(tok)
        return "".join(tokens)

    def encode_with_special(
        self,
        prompt: str,
        output_prefix: str,
        update: str,
        revised_continuation: str,
    ) -> dict[str, list[int]]:
        """Encode a full duplex sample into baseline-serialized token IDs."""
        full = (
            f"<BOS>{prompt}<SEP>{output_prefix}"
            f"<UPDATE>{update}<SEP>{revised_continuation}<EOS>"
        )
        return {
            "input_ids": self.encode(full),
            "prompt_len": 1 + len(self.encode(prompt)) + 1,  # BOS + prompt + SEP
        }

    def encode_structured(
        self,
        prompt: str,
        output_prefix: str,
        update: str,
        revised_continuation: str,
    ) -> dict[str, list[int]]:
        """Encode a duplex sample as separate streams for the workspace model."""
        return {
            "prompt_ids": self.encode(prompt),
            "prefix_ids": self.encode(output_prefix),
            "update_ids": self.encode(update),
            "continuation_ids": self.encode(revised_continuation),
        }
