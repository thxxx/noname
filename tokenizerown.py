import logging
import re
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_SENTRY_L = "\uE000"
_SENTRY_R = "\uE001"


class LibriTTSTokenizer:
    """
    Char-only tokenizer with special-token protection around normalization.
    """

    def __init__(
        self,
        token_file: str,
        *,
        special_tokens: Optional[List[str]] = None,
        lowercase: bool = True,
        use_normalize: bool = True,
        oov_policy: str = "skip",
        unk_token: Optional[str] = None,
    ):
        self.type = "char"
        self.has_tokens = False
        self.lowercase = lowercase
        self.use_normalize = use_normalize
        self.oov_policy = oov_policy
        self.unk_token = unk_token

        if self.use_normalize:
            try:
                import tacotron_cleaner.cleaners as _cleaners
                self._normalize_fn = _cleaners.custom_english_cleaners
            except Exception as ex:
                logger.warning(
                    f"{ex}\nFalling back to identity normalization. "
                    "Run `pip install espnet_tts_frontend` to enable cleaners."
                )
                self._normalize_fn = lambda s: s
        else:
            self._normalize_fn = lambda s: s

        self.token2id: Dict[str, int] = {}
        with open(token_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) != 2:
                    raise ValueError(f"Bad line in token file: {line!r}")
                token, idx_str = parts[0], parts[1]
                idx = int(idx_str)
                if token in self.token2id:
                    raise ValueError(f"Duplicate token in token file: {token}")
                self.token2id[token] = idx

        self.vocab_size = len(self.token2id)
        if "_" not in self.token2id:
            raise KeyError('Padding token "_" not found in token file.')
        self.pad_id = self.token2id["_"]

        if self.oov_policy == "use_unk":
            if not self.unk_token:
                raise ValueError("unk_token must be provided when oov_policy='use_unk'")
            if self.unk_token not in self.token2id:
                raise KeyError(f"unk_token {self.unk_token!r} not found in token file.")
            self.unk_id = self.token2id[self.unk_token]
        else:
            self.unk_id = None

        # 🔧 build reverse map
        self.id2token: Dict[int, str] = {v: k for k, v in self.token2id.items()}

        self.special_tokens = sorted(special_tokens or [], key=len, reverse=True)
        self._special_re = (
            re.compile("(" + "|".join(re.escape(t) for t in self.special_tokens) + ")")
            if self.special_tokens
            else None
        )

        self.has_tokens = True

    # ---------- Public API ----------

    def texts_to_token_ids(self, texts: List[str]) -> List[List[int]]:
        tokens_list = self.texts_to_tokens(texts)
        return self.tokens_to_token_ids(tokens_list)

    def texts_to_tokens(self, texts: List[str]) -> List[List[str]]:
        out: List[List[str]] = []
        for text in texts:
            protected, idx_map = self._protect_specials(text)
            # norm은 사용 안함 (요청 코드 유지)
            restored = self._restore_specials(protected, idx_map)
            tokens = self._split_char_with_specials(restored)
            out.append(tokens)
        return out

    def tokens_to_token_ids(self, tokens_list: List[List[str]]) -> List[List[int]]:
        assert self.has_tokens, "Tokenizer not initialized with tokens."
        results: List[List[int]] = []
        for tokens in tokens_list:
            ids: List[int] = []
            for t in tokens:
                if t in self.token2id:
                    ids.append(self.token2id[t])
                else:
                    if self.oov_policy == "skip":
                        logger.debug(f"Skip OOV token: {t!r}")
                        continue
                    elif self.oov_policy == "error":
                        raise KeyError(f"OOV token encountered: {t!r}")
                    elif self.oov_policy == "use_unk":
                        ids.append(self.unk_id)  # type: ignore[arg-type]
                    else:
                        raise ValueError(f"Unknown oov_policy: {self.oov_policy}")
            results.append(ids)
        return results

    # 🔧 NEW: ids -> tokens
    def token_ids_to_tokens(
        self,
        ids_list: List[List[int]],
        *,
        skip_pad: bool = True,
        error_on_unknown_id: bool = False,
    ) -> List[List[str]]:
        """
        Reverse of tokens_to_token_ids.
        - skip_pad: drop padding IDs.
        - error_on_unknown_id: raise if an ID is not in id2token; otherwise skip it.
        """
        results: List[List[str]] = []
        for ids in ids_list:
            toks: List[str] = []
            for i in ids:
                if skip_pad and i == self.pad_id:
                    continue
                tok = self.id2token.get(i)
                if tok is None:
                    if error_on_unknown_id:
                        raise KeyError(f"Unknown token id: {i}")
                    logger.debug(f"Skip unknown id: {i}")
                    continue
                toks.append(tok)
            results.append(toks)
        return results

    # 🔧 NEW: tokens -> text
    def tokens_to_texts(self, tokens_list: List[List[str]]) -> List[str]:
        """
        Simple detokenization for char-level tokens with special tokens preserved.
        Note: original casing is not recoverable if `lowercase=True` was used.
        """
        return ["".join(toks) for toks in tokens_list]

    # 🔧 NEW: ids -> text (convenience)
    def token_ids_to_texts(
        self,
        ids_list: List[List[int]],
        *,
        skip_pad: bool = True,
        error_on_unknown_id: bool = False,
    ) -> List[str]:
        toks = self.token_ids_to_tokens(
            ids_list, skip_pad=skip_pad, error_on_unknown_id=error_on_unknown_id
        )
        return self.tokens_to_texts(toks)

    # ---------- Token table management ----------

    def add_token(self, token: str, token_id: Optional[int] = None):
        if token in self.token2id:
            raise ValueError(f"Token already exists: {token!r}")
        if token_id is None:
            token_id = self.vocab_size
        elif token_id in self.token2id.values():
            raise ValueError(f"Token id already used: {token_id}")
        self.token2id[token] = token_id
        self.id2token[token_id] = token  # 🔧 keep reverse map in sync
        self.vocab_size = len(self.token2id)

    # ---------- Internal helpers ----------

    def _protect_specials(self, text: str):
        if not self._special_re:
            return text, {}
        idx_map: Dict[str, str] = {}
        counter = 0

        def repl(m):
            nonlocal counter
            tok = m.group(0)
            key = f"{_SENTRY_L}{counter}{_SENTRY_R}"
            idx_map[key] = tok
            counter += 1
            return key

        protected = self._special_re.sub(repl, text)
        return protected, idx_map

    def _restore_specials(self, text: str, idx_map: Dict[str, str]):
        if not idx_map:
            return text
        for key, tok in idx_map.items():
            text = text.replace(key, tok)
        return text

    def _split_char_with_specials(self, text: str) -> List[str]:
        if not self._special_re:
            return list(text.lower() if self.lowercase else text)

        tokens: List[str] = []
        pos = 0
        for m in self._special_re.finditer(text):
            chunk = text[pos:m.start()]
            if chunk:
                tokens.extend(list(chunk.lower() if self.lowercase else chunk))
            tokens.append(m.group(0))
            pos = m.end()

        tail = text[pos:]
        if tail:
            tokens.extend(list(tail.lower() if self.lowercase else tail))
        return tokens
