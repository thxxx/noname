# Copyright      2023-2024  Xiaomi Corp.        (authors: Zengwei Yao
#                                                         Han Zhu,
#                                                         Wei Kang)
#
# See ../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import re
from abc import ABC, abstractmethod
from functools import reduce
from typing import Dict, List, Optional

import jieba
from lhotse import CutSet
from pypinyin import Style, lazy_pinyin
from pypinyin.contrib.tone_convert import to_finals_tone3, to_initials

try:
    from piper_phonemize import phonemize_espeak
except Exception as ex:
    raise RuntimeError(
        f"{ex}\nPlease run\n"
        "pip install piper_phonemize -f \
            https://k2-fsa.github.io/icefall/piper_phonemize.html"
    )

import re
from abc import ABC, abstractmethod

import cn2an
import inflect


class TextNormalizer(ABC):
    """Abstract base class for text normalization, defining common interface."""

    @abstractmethod
    def normalize(self, text: str) -> str:
        """Normalize text."""
        raise NotImplementedError


class EnglishTextNormalizer(TextNormalizer):
    """
    A class to handle preprocessing of English text including normalization. Following:
    https://github.com/espnet/espnet_tts_frontend/blob/master/tacotron_cleaner/cleaners.py
    """

    def __init__(self):
        # List of (regular expression, replacement) pairs for abbreviations:
        self._abbreviations = [
            (re.compile("\\b%s\\b" % x[0], re.IGNORECASE), x[1])
            for x in [
                ("mrs", "misess"),
                ("mr", "mister"),
                ("dr", "doctor"),
                ("st", "saint"),
                ("co", "company"),
                ("jr", "junior"),
                ("maj", "major"),
                ("gen", "general"),
                ("drs", "doctors"),
                ("rev", "reverend"),
                ("lt", "lieutenant"),
                ("hon", "honorable"),
                ("sgt", "sergeant"),
                ("capt", "captain"),
                ("esq", "esquire"),
                ("ltd", "limited"),
                ("col", "colonel"),
                ("ft", "fort"),
                ("etc", "et cetera"),
                ("btw", "by the way"),
            ]
        ]

        self._inflect = inflect.engine()
        self._comma_number_re = re.compile(r"([0-9][0-9\,]+[0-9])")
        self._decimal_number_re = re.compile(r"([0-9]+\.[0-9]+)")
        self._percent_number_re = re.compile(r"([0-9\.\,]*[0-9]+%)")
        self._pounds_re = re.compile(r"£([0-9\,]*[0-9]+)")
        self._dollars_re = re.compile(r"\$([0-9\.\,]*[0-9]+)")
        self._fraction_re = re.compile(r"([0-9]+)/([0-9]+)")
        self._ordinal_re = re.compile(r"[0-9]+(st|nd|rd|th)")
        self._number_re = re.compile(r"[0-9]+")
        self._whitespace_re = re.compile(r"\s+")

    def normalize(self, text: str) -> str:
        """Custom pipeline for English text,
        including number and abbreviation expansion."""
        text = self.expand_abbreviations(text)
        text = self.normalize_numbers(text)

        return text

    def fraction_to_words(self, numerator, denominator):
        if numerator == 1 and denominator == 2:
            return " one half "
        if numerator == 1 and denominator == 4:
            return " one quarter "
        if denominator == 2:
            return " " + self._inflect.number_to_words(numerator) + " halves "
        if denominator == 4:
            return " " + self._inflect.number_to_words(numerator) + " quarters "
        return (
            " "
            + self._inflect.number_to_words(numerator)
            + " "
            + self._inflect.ordinal(self._inflect.number_to_words(denominator))
            + " "
        )

    def _remove_commas(self, m):
        return m.group(1).replace(",", "")

    def _expand_dollars(self, m):
        match = m.group(1)
        parts = match.split(".")
        if len(parts) > 2:
            return " " + match + " dollars "  # Unexpected format
        dollars = int(parts[0]) if parts[0] else 0
        cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
        if dollars and cents:
            dollar_unit = "dollar" if dollars == 1 else "dollars"
            cent_unit = "cent" if cents == 1 else "cents"
            return " %s %s, %s %s " % (dollars, dollar_unit, cents, cent_unit)
        elif dollars:
            dollar_unit = "dollar" if dollars == 1 else "dollars"
            return " %s %s " % (dollars, dollar_unit)
        elif cents:
            cent_unit = "cent" if cents == 1 else "cents"
            return " %s %s " % (cents, cent_unit)
        else:
            return " zero dollars "

    def _expand_fraction(self, m):
        numerator = int(m.group(1))
        denominator = int(m.group(2))
        return self.fraction_to_words(numerator, denominator)

    def _expand_decimal_point(self, m):
        return m.group(1).replace(".", " point ")

    def _expand_percent(self, m):
        return m.group(1).replace("%", " percent ")

    def _expand_ordinal(self, m):
        return " " + self._inflect.number_to_words(m.group(0)) + " "

    def _expand_number(self, m):
        num = int(m.group(0))
        if num > 1000 and num < 3000:
            if num == 2000:
                return " two thousand "
            elif num > 2000 and num < 2010:
                return " two thousand " + self._inflect.number_to_words(num % 100) + " "
            elif num % 100 == 0:
                return " " + self._inflect.number_to_words(num // 100) + " hundred "
            else:
                return (
                    " "
                    + self._inflect.number_to_words(
                        num, andword="", zero="oh", group=2
                    ).replace(", ", " ")
                    + " "
                )
        else:
            return " " + self._inflect.number_to_words(num, andword="") + " "

    def normalize_numbers(self, text):
        text = re.sub(self._comma_number_re, self._remove_commas, text)
        text = re.sub(self._pounds_re, r"\1 pounds", text)
        text = re.sub(self._dollars_re, self._expand_dollars, text)
        text = re.sub(self._fraction_re, self._expand_fraction, text)
        text = re.sub(self._decimal_number_re, self._expand_decimal_point, text)
        text = re.sub(self._percent_number_re, self._expand_percent, text)
        text = re.sub(self._ordinal_re, self._expand_ordinal, text)
        text = re.sub(self._number_re, self._expand_number, text)
        return text

    def expand_abbreviations(self, text):
        for regex, replacement in self._abbreviations:
            text = re.sub(regex, replacement, text)
        return text


class ChineseTextNormalizer(TextNormalizer):
    """
    A class to handle preprocessing of Chinese text including normalization.
    """

    def normalize(self, text: str) -> str:
        """Normalize text."""
        # Convert numbers to Chinese
        text = cn2an.transform(text, "an2cn")
        return text

jieba.default_logger.setLevel(logging.INFO)
class KoreanTextNormalizer:
    """
    최소한의 전처리: 공백/문장부호 정규화, 한글 호환 자모를 표준 자모로 치환(필요시).
    필요하다면 숫자->한국어(예: '5'->'오') 치환 규칙을 여기에 확장하세요.
    """
    def normalize(self, text: str) -> str:
        # 중국어와 동일한 문장부호 맵을 재사용
        text = text.replace("，", ",").replace("。", ".").replace("！", "!")
        text = text.replace("？", "?").replace("；", ";").replace("：", ":")
        text = text.replace("、", ",").replace("‘", "'").replace("’", "'")
        text = text.replace("“", '"').replace("”", '"')
        text = text.replace("⋯", "…").replace("···", "…").replace("・・・", "…")
        text = text.replace("...", "…")
        # 연속 공백 정규화
        text = re.sub(r"\s+", " ", text).strip()
        return text
# ---- 한글 감지 ----
def is_korean_char(ch: str) -> bool:
    # 음절(U+AC00–U+D7A3)만 대상으로 분해 (자모 영역은 그대로 토큰으로도 쓸 수 있음)
    return "\uAC00" <= ch <= "\uD7A3"

# ---- 한글 음절 -> (초성, 중성, 종성) 분해 ----
_CHO = ['ㄱ','ㄲ','ㄴ','ㄷ','ㄸ','ㄹ','ㅁ','ㅂ','ㅃ','ㅅ','ㅆ','ㅇ','ㅈ','ㅉ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']
_JUNG = ['ㅏ','ㅐ','ㅑ','ㅒ','ㅓ','ㅔ','ㅕ','ㅖ','ㅗ','ㅘ','ㅙ','ㅚ','ㅛ','ㅜ','ㅝ','ㅞ','ㅟ','ㅠ','ㅡ','ㅢ','ㅣ']
_JONG = ['','ㄱ','ㄲ','ㄳ','ㄴ','ㄵ','ㄶ','ㄷ','ㄹ','ㄺ','ㄻ','ㄼ','ㄽ','ㄾ','ㄿ','ㅀ','ㅁ','ㅂ','ㅄ','ㅅ','ㅆ','ㅇ','ㅈ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']

def decompose_hangul_syllable(ch: str):
    """AC00 기반 분해. 반환은 (초성 or None, 중성 or None, 종성 or None)"""
    if not is_korean_char(ch):
        return (None, None, None)
    code = ord(ch) - 0xAC00
    cho = code // 588
    jung = (code % 588) // 28
    jong = code % 28
    return (_CHO[cho], _JUNG[jung], _JONG[jong] if _JONG[jong] != '' else None)


class Tokenizer(ABC):
    """Abstract base class for tokenizers, defining common interface."""

    @abstractmethod
    def texts_to_token_ids(self, texts: List[str]) -> List[List[int]]:
        """Convert list of texts to list of token id sequences."""
        raise NotImplementedError

    @abstractmethod
    def texts_to_tokens(self, texts: List[str]) -> List[List[str]]:
        """Convert list of texts to list of token sequences."""
        raise NotImplementedError

    @abstractmethod
    def tokens_to_token_ids(self, tokens: List[List[str]]) -> List[List[int]]:
        """Convert list of token sequences to list of token id sequences."""
        raise NotImplementedError


class SimpleTokenizer(Tokenizer):
    """The simplpest tokenizer, treat every character as a token,
    without text normalization.
    """

    def __init__(self, token_file: Optional[str] = None):
        """
        Args:
          tokens: the file that contains information that maps tokens to ids,
            which is a text file with '{token}\t{token_id}' per line.
        """
        # Parse token file
        self.has_tokens = False
        if token_file is None:
            logging.debug(
                "Initialize Tokenizer without tokens file, \
                will fail when map to ids."
            )
            return
        self.token2id: Dict[str, int] = {}
        with open(token_file, "r", encoding="utf-8") as f:
            for line in f.readlines():
                info = line.rstrip().split("\t")
                token, id = info[0], int(info[1])
                assert token not in self.token2id, token
                self.token2id[token] = id
        self.pad_id = self.token2id["_"]  # padding
        self.vocab_size = len(self.token2id)
        self.has_tokens = True

    def texts_to_token_ids(
        self,
        texts: List[str],
    ) -> List[List[int]]:
        return self.tokens_to_token_ids(self.texts_to_tokens(texts))

    def texts_to_tokens(
        self,
        texts: List[str],
    ) -> List[List[str]]:
        tokens_list = [list(texts[i]) for i in range(len(texts))]
        return tokens_list

    def tokens_to_token_ids(
        self,
        tokens_list: List[List[str]],
    ) -> List[List[int]]:
        assert self.has_tokens, "Please initialize Tokenizer with a tokens file."

        token_ids_list = []

        for tokens in tokens_list:
            token_ids = []
            for t in tokens:
                if t not in self.token2id:
                    logging.debug(f"Skip OOV {t}")
                    continue
                token_ids.append(self.token2id[t])

            token_ids_list.append(token_ids)

        return token_ids_list


class EspeakTokenizer(Tokenizer):
    """A simple tokenizer with Espeak g2p function."""

    def __init__(self, token_file: Optional[str] = None, lang: str = "en-us"):
        """
        Args:
          tokens: the file that contains information that maps tokens to ids,
            which is a text file with '{token}\t{token_id}' per line.
          lang: the language identifier, see
            https://github.com/rhasspy/espeak-ng/blob/master/docs/languages.md
        """
        # Parse token file
        self.has_tokens = False
        self.lang = lang
        if token_file is None:
            logging.debug(
                "Initialize Tokenizer without tokens file, \
                will fail when map to ids."
            )
            return
        self.token2id: Dict[str, int] = {}
        with open(token_file, "r", encoding="utf-8") as f:
            for line in f.readlines():
                info = line.rstrip().split("\t")
                token, id = info[0], int(info[1])
                assert token not in self.token2id, token
                self.token2id[token] = id
        self.pad_id = self.token2id["_"]  # padding
        self.vocab_size = len(self.token2id)
        self.has_tokens = True

    def g2p(self, text: str) -> List[str]:
        try:
            tokens = phonemize_espeak(text, self.lang)
            tokens = reduce(lambda x, y: x + y, tokens)
            return tokens
        except Exception as ex:
            logging.warning(f"Tokenization of {self.lang} texts failed: {ex}")
            return []

    def texts_to_token_ids(
        self,
        texts: List[str],
    ) -> List[List[int]]:
        return self.tokens_to_token_ids(self.texts_to_tokens(texts))

    def texts_to_tokens(
        self,
        texts: List[str],
    ) -> List[List[str]]:
        tokens_list = [self.g2p(texts[i]) for i in range(len(texts))]
        return tokens_list

    def tokens_to_token_ids(
        self,
        tokens_list: List[List[str]],
    ) -> List[List[int]]:
        assert self.has_tokens, "Please initialize Tokenizer with a tokens file."

        token_ids_list = []

        for tokens in tokens_list:
            token_ids = []
            for t in tokens:
                if t not in self.token2id:
                    logging.debug(f"Skip OOV {t}")
                    continue
                token_ids.append(self.token2id[t])

            token_ids_list.append(token_ids)

        return token_ids_list


class EmiliaTokenizer(Tokenizer):
    def __init__(self, token_file: Optional[str] = None, token_type="phone"):
        """
        Args:
          tokens: the file that contains information that maps tokens to ids,
            which is a text file with '{token}\t{token_id}' per line.
        """
        assert (
            token_type == "phone"
        ), f"Only support phone tokenizer for Emilia, but get {token_type}."

        self.english_normalizer = EnglishTextNormalizer()
        self.chinese_normalizer = ChineseTextNormalizer()
        self.korean_normalizer  = KoreanTextNormalizer()  # <-- 추가

        self.has_tokens = False
        if token_file is None:
            logging.debug(
                "Initialize Tokenizer without tokens file, \
                    will fail when map to ids."
            )
            return
        self.token2id: Dict[str, int] = {}
        with open(token_file, "r", encoding="utf-8") as f:
            for line in f.readlines():
                info = line.rstrip().split("\t")
                token, id = info[0], int(info[1])
                assert token not in self.token2id, token
                self.token2id[token] = id
        self.pad_id = self.token2id["_"]  # padding

        self.vocab_size = len(self.token2id)
        self.has_tokens = True
    
        # 토큰 파일이 없는 경우는 스킵
        if getattr(self, "has_tokens", False):
            self.id2token: Dict[int, str] = {v: k for k, v in self.token2id.items()}
        else:
            self.id2token = {}

    def texts_to_token_ids(
        self,
        texts: List[str],
    ) -> List[List[int]]:
        return self.tokens_to_token_ids(self.texts_to_tokens(texts))

    # ---- 기존 preprocess_text는 중국어 구두점 맵만 수행 ----
    def preprocess_text(self, text: str) -> str:
        return self.map_punctuations(text)

    def texts_to_tokens(self, texts: List[str]) -> List[List[str]]:
        for i in range(len(texts)):
            texts[i] = self.preprocess_text(texts[i])

        phoneme_list = []
        for text in texts:
            segments = self.get_segment(text)
            all_phoneme = []
            for seg_text, seg_lang in segments:
                if seg_lang == "zh":
                    phoneme = self.tokenize_ZH(seg_text)
                elif seg_lang == "en":
                    phoneme = self.tokenize_EN(seg_text)
                elif seg_lang == "ko":  # <-- 한국어 분기
                    phoneme = self.tokenize_KO(seg_text)
                elif seg_lang == "pinyin":
                    phoneme = self.tokenize_pinyin(seg_text)
                elif seg_lang == "tag":
                    phoneme = [seg_text]
                else:
                    logging.warning(f"No supported language; skipping: {(seg_text, seg_lang)}")
                    continue
                all_phoneme += phoneme
            phoneme_list.append(all_phoneme)
        return phoneme_list
# ----------------------------
    # IDs -> Tokens -> Text
    # ----------------------------
    def token_ids_to_tokens(self, token_ids_list: List[List[int]]) -> List[List[str]]:
        """
        Map a batch of token-id sequences back to token strings.
        OOV id는 스킵.
        """
        assert hasattr(self, "id2token"), "id2token not built. Call __post_init_id2token in __init__."
        out: List[List[str]] = []
        for ids in token_ids_list:
            toks: List[str] = []
            for i in ids:
                t = self.id2token.get(i, None)
                if t is None:
                    logging.debug(f"Skip unknown id {i}")
                    continue
                toks.append(t)
            out.append(toks)
        return out

    def token_ids_to_texts(self, token_ids_list: List[List[int]]) -> List[str]:
        """
        Batch: token ids -> readable text (best-effort detokenization).
        """
        tokens_list = self.token_ids_to_tokens(token_ids_list)
        return [self.tokens_to_text(tokens) for tokens in tokens_list]
    
    def tokens_to_text(self, tokens: List[str]) -> str:
        """
        Best-effort detokenizer.
        - KO: ㄱ0 + ㅏ (+ ㄴ) -> '간' 식으로 합성
        - ZH pinyin: sh0 + uang3 -> "<shuang3>" 로 결합 (가독성 목적)
        - EN/others: 공백 + 간단한 구두점 공백 정리
        """
        pieces: List[str] = []
        i = 0
        while i < len(tokens):
            tok = tokens[i]

            # 태그는 그대로
            if self.is_tag(tok):
                pieces.append(tok)
                i += 1
                continue

            # ----- Korean 조합: 초성0 + 중성 (+ 종성)
            if self._is_korean_initial(tok):
                cho = tok[:-1]  # remove trailing '0'
                jung = None
                jong = None

                # lookahead for jung
                if i + 1 < len(tokens) and self._is_korean_vowel(tokens[i + 1]):
                    jung = tokens[i + 1]
                    i_advance = 2

                    # optional jong
                    if i + 2 < len(tokens) and self._is_korean_final(tokens[i + 2]):
                        # 종성 후보가 실제로 합성 가능한 조합일 때만 사용
                        candidate_jong = tokens[i + 2]
                        if self._can_compose_korean(cho, jung, candidate_jong):
                            jong = candidate_jong
                            i_advance = 3

                    ch = self._compose_hangul(cho, jung, jong)
                    pieces.append(ch)
                    i += i_advance
                    continue
                else:
                    # 중성이 없으면 초성 기호 자체를 출력(정보 손실 보호)
                    pieces.append(cho)
                    i += 1
                    continue

            # ----- Chinese pinyin 재결합: initial0 + finals(tone)
            if self._looks_pinyin_initial(tok):
                if i + 1 < len(tokens) and self._looks_pinyin_finals(tokens[i + 1]):
                    p = tok[:-1] + tokens[i + 1]  # 'sh0' + 'uang3' -> 'shuang3'
                    pieces.append(f"<{p}>")
                    i += 2
                    continue
                else:
                    # finals가 없으면 그냥 출력
                    pieces.append(tok)
                    i += 1
                    continue

            # finals 단독으로 온 pinyin도 <>로 감쌈
            if self._looks_pinyin_finals(tok):
                pieces.append(f"<{tok}>")
                i += 1
                continue

            # 그 외: 그냥 추가
            pieces.append(tok)
            i += 1

        # 간단한 공백/구두점 정리
        text = " ".join(pieces)
        text = (
            text.replace(" ,", ",")
                .replace(" .", ".")
                .replace(" !", "!")
                .replace(" ?", "?")
                .replace(" ;", ";")
                .replace(" :", ":")
        )
        # 중괄호/대괄호/꺾쇠 내부 앞뒤 공백 제거
        text = re.sub(r"\s+([\]\)])", r"\1", text)
        text = re.sub(r"([\[\(])\s+", r"\1", text)
        text = re.sub(r"\s+(>)", r"\1", text)
        text = re.sub(r"(<)\s+", r"\1", text)
        text = re.sub(r"\s{2,}", " ", text).strip()
        return text
    
    # ---- 한국어 토크나이저 ----
    def tokenize_KO(self, text: str) -> List[str]:
        try:
            text = self.korean_normalizer.normalize(text)
            tokens: List[str] = []
            for ch in text:
                    tokens.append(ch)
            return tokens
        except Exception as ex:
            logging.warning(f"Tokenization of Korean texts failed: {ex}")
            return []

    # ---- 세그멘테이션 보강: 한국어 타입 판별 ----
    def get_segment(self, text: str) -> List[str]:
        segments = []
        types = []
        temp_seg = ""
        temp_lang = ""

        _part_pattern = re.compile(r"[<[].*?[>\]]|.")
        text_parts = _part_pattern.findall(text)

        for part in text_parts:
            if self.is_pinyin(part):
                types.append("zh")  # pinyin은 중국어 처리 루틴을 타므로 'zh'로 유지 (아래 split에서 'pinyin'으로 재태깅)
            elif self.is_chinese(part):
                types.append("zh")
            elif self.is_korean(part):                # <-- 추가
                types.append("ko")
            elif self.is_alphabet(part):
                types.append("en")
            else:
                types.append("other")

        assert len(types) == len(text_parts)

        for i in range(len(types)):
            if i == 0:
                temp_seg += text_parts[i]
                temp_lang = types[i]
            else:
                if temp_lang == "other":
                    temp_seg += text_parts[i]
                    temp_lang = types[i]
                else:
                    if types[i] in [temp_lang, "other"]:
                        temp_seg += text_parts[i]
                    else:
                        segments.append((temp_seg, temp_lang))
                        temp_seg = text_parts[i]
                        temp_lang = types[i]
        segments.append((temp_seg, temp_lang))

        # 기존처럼 <>/[] 처리
        segments = self.split_segments(segments)
        return segments

    # ---- 헬퍼: 한국어 판별 ----
    def is_korean(self, part: str) -> bool:
        # part는 한 글자 또는 <>/[] 포함 문자열일 수 있음. 여기서는 한 글자 기준으로만 판별
        return len(part) == 1 and is_korean_char(part)

    def tokens_to_token_ids(
        self,
        tokens_list: List[List[str]],
    ) -> List[List[int]]:
        assert self.has_tokens, "Please initialize Tokenizer with a tokens file."
        token_ids_list = []

        for tokens in tokens_list:
            token_ids = []
            for t in tokens:
                if t not in self.token2id:
                    logging.debug(f"Skip OOV {t}")
                    continue
                token_ids.append(self.token2id[t])

            token_ids_list.append(token_ids)

        return token_ids_list

    def tokenize_ZH(self, text: str) -> List[str]:
        try:
            text = self.chinese_normalizer.normalize(text)
            segs = list(jieba.cut(text))
            full = lazy_pinyin(
                segs,
                style=Style.TONE3,
                tone_sandhi=True,
                neutral_tone_with_five=True,
            )
            phones = []
            for x in full:
                # valid pinyin (in tone3 style) is alphabet + 1 number in [1-5].
                if not (x[0:-1].isalpha() and x[-1] in ("1", "2", "3", "4", "5")):
                    phones.append(x)
                    continue
                else:
                    phones.extend(self.seperate_pinyin(x))
            return phones
        except Exception as ex:
            logging.warning(f"Tokenization of Chinese texts failed: {ex}")
            return []

    def tokenize_EN(self, text: str) -> List[str]:
        try:
            text = self.english_normalizer.normalize(text)
            tokens = phonemize_espeak(text, "en-us")
            tokens = reduce(lambda x, y: x + y, tokens)
            return tokens
        except Exception as ex:
            logging.warning(f"Tokenization of English texts failed: {ex}")
            return []

    def tokenize_pinyin(self, text: str) -> List[str]:
        try:
            assert text.startswith("<") and text.endswith(">")
            text = text.lstrip("<").rstrip(">")
            # valid pinyin (in tone3 style) is alphabet + 1 number in [1-5].
            if not (text[0:-1].isalpha() and text[-1] in ("1", "2", "3", "4", "5")):
                logging.warning(
                    f"Strings enclosed with <> should be pinyin, \
                    but got: {text}. Skipped it. "
                )
                return []
            else:
                return self.seperate_pinyin(text)
        except Exception as ex:
            logging.warning(f"Tokenize pinyin failed: {ex}")
            return []

    def seperate_pinyin(self, text: str) -> List[str]:
        """
        Separate pinyin into initial and final
        """
        pinyins = []
        initial = to_initials(text, strict=False)
        # don't want to share tokens with espeak tokens,
        # so use tone3 style
        final = to_finals_tone3(
            text,
            strict=False,
            neutral_tone_with_five=True,
        )
        if initial != "":
            # don't want to share tokens with espeak tokens,
            # so add a '0' after each initial
            pinyins.append(initial + "0")
        if final != "":
            pinyins.append(final)
        return pinyins

    def map_punctuations(self, text):
        text = text.replace("，", ",")
        text = text.replace("。", ".")
        text = text.replace("！", "!")
        text = text.replace("？", "?")
        text = text.replace("；", ";")
        text = text.replace("：", ":")
        text = text.replace("、", ",")
        text = text.replace("‘", "'")
        text = text.replace("“", '"')
        text = text.replace("”", '"')
        text = text.replace("’", "'")
        text = text.replace("⋯", "…")
        text = text.replace("···", "…")
        text = text.replace("・・・", "…")
        text = text.replace("...", "…")
        return text

    def split_segments(self, segments):
        """
        split segments into smaller parts if special strings enclosed by [] or <>
        are found, where <> denotes pinyin strings, [] denotes other special strings.

        Args:
            segments (list): A list of tuples where each tuple contains:
                - temp_seg (str): The text segment to be split.
                - temp_lang (str): The language code associated with the segment.

        Returns:
            list: A list of smaller segments.
        """
        result = []
        for temp_seg, temp_lang in segments:
            parts = re.split(r"([<[].*?[>\]])", temp_seg)
            for part in parts:
                if not part:
                    continue
                if self.is_pinyin(part):
                    result.append((part, "pinyin"))
                elif self.is_tag(part):
                    result.append((part, "tag"))
                else:
                    result.append((part, temp_lang))
        return result

    def is_chinese(self, char: str) -> bool:
        if char >= "\u4e00" and char <= "\u9fa5":
            return True
        else:
            return False

    def is_alphabet(self, char: str) -> bool:
        if (char >= "\u0041" and char <= "\u005a") or (
            char >= "\u0061" and char <= "\u007a"
        ):
            return True
        else:
            return False

    def is_pinyin(self, part: str) -> bool:
        if part.startswith("<") and part.endswith(">"):
            return True
        else:
            return False

    def is_tag(self, part: str) -> bool:
        if part.startswith("[") and part.endswith("]"):
            return True
        else:
            return False
    
    _KO_CHO = ['ㄱ','ㄲ','ㄴ','ㄷ','ㄸ','ㄹ','ㅁ','ㅂ','ㅃ','ㅅ','ㅆ','ㅇ','ㅈ','ㅉ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']
    _KO_JUNG = ['ㅏ','ㅐ','ㅑ','ㅒ','ㅓ','ㅔ','ㅕ','ㅖ','ㅗ','ㅘ','ㅙ','ㅚ','ㅛ','ㅜ','ㅝ','ㅞ','ㅟ','ㅠ','ㅡ','ㅢ','ㅣ']
    _KO_JONG = [None,'ㄱ','ㄲ','ㄳ','ㄴ','ㄵ','ㄶ','ㄷ','ㄹ','ㄺ','ㄻ','ㄼ','ㄽ','ㄾ','ㄿ','ㅀ','ㅁ','ㅂ','ㅄ','ㅅ','ㅆ','ㅇ','ㅈ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']

    def _is_korean_initial(self, t: str) -> bool:
        # 초성은 항상 '0' suffix를 붙였다는 규칙 사용 (예: 'ㄱ0')
        return len(t) >= 2 and t.endswith("0") and (t[:-1] in self._KO_CHO)

    def _is_korean_vowel(self, t: str) -> bool:
        return t in self._KO_JUNG

    def _is_korean_final(self, t: str) -> bool:
        # 종성은 None을 제외한 리스트 멤버(복합종성 포함)
        return t in self._KO_JONG[1:]

    def _can_compose_korean(self, cho: str, jung: str, jong: Optional[str]) -> bool:
        return (cho in self._KO_CHO) and (jung in self._KO_JUNG) and (jong in self._KO_JONG if (jong is None or isinstance(jong, str)) else False)

    def _compose_hangul(self, cho: str, jung: str, jong: Optional[str]) -> str:
        """
        초성/중성/(종성) -> 완성형 한글 합성
        """
        try:
            L = self._KO_CHO.index(cho)
            V = self._KO_JUNG.index(jung)
            T = self._KO_JONG.index(jong)  # None -> 0
            code = 0xAC00 + (L * 21 * 28) + (V * 28) + T
            return chr(code)
        except ValueError:
            # 합성 실패 시 가능한 것만 이어붙임
            return cho + jung + (jong or "")
    
    # pinyin 규칙: initial0, finals(tone digit 1-5)
    _PINYIN_INITIALS = set([
        # 표준 한자음 초성 (lazy_pinyin initials)
        "b","p","m","f","d","t","n","l","g","k","h","j","q","x","zh","ch","sh","r","z","c","s","y","w"
    ])

    def _looks_pinyin_initial(self, t: str) -> bool:
        return t.endswith("0") and (t[:-1] in self._PINYIN_INITIALS)

    def _looks_pinyin_finals(self, t: str) -> bool:
        # tone3 style finals: 알파벳 + 마지막 한 자리 숫자(1~5)
        return (len(t) >= 2) and t[:-1].isalpha() and (t[-1] in "12345")


class DialogTokenizer(EmiliaTokenizer):
    def __init__(self, token_file: Optional[str] = None, token_type="phone"):
        super().__init__(token_file=token_file, token_type=token_type)
        if token_file:
            self.spk_a_id = self.token2id["[S1]"]
            self.spk_b_id = self.token2id["[S2]"]

    def preprocess_text(
        self,
        text: str,
    ) -> str:
        text = re.sub(r"\s*(\[S[12]\])\s*", r"\1", text)
        text = self.map_punctuations(text)
        return text


class LibriTTSTokenizer(Tokenizer):
    def __init__(self, token_file: Optional[str] = None, token_type="char"):
        """
        Args:
          type: the type of tokenizer, e.g., bpe, char, phone.
          tokens: the file that contains information that maps tokens to ids,
            which is a text file with '{token}\t{token_id}' per line if type is
            char or phone, otherwise it is a bpe_model file.
        """
        self.type = token_type
        assert token_type in ["bpe", "char", "phone"]
        try:
            import tacotron_cleaner.cleaners
        except Exception as ex:
            raise RuntimeError(f"{ex}\nPlease run\n" "pip install espnet_tts_frontend")

        self.normalize = tacotron_cleaner.cleaners.custom_english_cleaners

        self.has_tokens = False
        if token_file is None:
            logging.debug(
                "Initialize Tokenizer without tokens file, \
                will fail when map to ids."
            )
            return
        if token_type == "bpe":
            import sentencepiece as spm

            self.sp = spm.SentencePieceProcessor()
            self.sp.load(token_file)
            self.pad_id = self.sp.piece_to_id("<pad>")
            self.vocab_size = self.sp.get_piece_size()
        else:
            self.token2id: Dict[str, int] = {}
            with open(token_file, "r", encoding="utf-8") as f:
                for line in f.readlines():
                    info = line.rstrip().split("\t")
                    token, id = info[0], int(info[1])
                    assert token not in self.token2id, token
                    self.token2id[token] = id
            self.pad_id = self.token2id["_"]  # padding
            self.vocab_size = len(self.token2id)
        self.has_tokens = True

    def texts_to_token_ids(
        self,
        texts: List[str],
    ) -> List[List[int]]:
        if self.type == "bpe":
            for i in range(len(texts)):
                texts[i] = self.normalize(texts[i])
            return self.sp.encode(texts)
        else:
            return self.tokens_to_token_ids(self.texts_to_tokens(texts))

    def texts_to_tokens(
        self,
        texts: List[str],
    ) -> List[List[str]]:
        # for i in range(len(texts)):
        #     texts[i] = self.normalize(texts[i])

        if self.type == "char":
            tokens_list = [list(texts[i].lower()) for i in range(len(texts))]
        elif self.type == "phone":
            tokens_list = [phonemize_espeak(texts[i].lower(), "en-us") for i in range(len(texts))]
        elif self.type == "bpe":
            tokens_list = self.sp.encode(texts, out_type=str)

        return tokens_list

    def tokens_to_token_ids(
        self,
        tokens_list: List[List[str]],
    ) -> List[List[int]]:
        assert self.has_tokens, "Please initialize Tokenizer with a tokens file."

        assert self.type != "bpe", "BPE tokenizer does not support this function."

        token_ids_list = []

        for tokens in tokens_list:
            token_ids = []
            for t in tokens:
                if t not in self.token2id:
                    logging.debug(f"Skip OOV {t}")
                    continue
                token_ids.append(self.token2id[t])

            token_ids_list.append(token_ids)

        return token_ids_list


def add_tokens(cut_set: CutSet, tokenizer: str, lang: str):
    if tokenizer == "emilia":
        tokenizer = EmiliaTokenizer()
    elif tokenizer == "espeak":
        tokenizer = EspeakTokenizer(lang=lang)
    elif tokenizer == "dialog":
        tokenizer = DialogTokenizer()
    elif tokenizer == "libritts":
        tokenizer = LibriTTSTokenizer()
    elif tokenizer == "simple":
        tokenizer = SimpleTokenizer()
    else:
        raise ValueError(f"Unsupported tokenizer: {tokenizer}.")

    def _prepare_cut(cut):
        # Each cut only contains one supervision
        assert len(cut.supervisions) == 1, (len(cut.supervisions), cut)
        text = cut.supervisions[0].text
        tokens = tokenizer.texts_to_tokens([text])[0]
        cut.supervisions[0].tokens = tokens
        return cut

    cut_set = cut_set.map(_prepare_cut)
    return cut_set


if __name__ == "__main__":
    text = (
        "我们是5年小米人,是吗? Yes I think so! "
        "mr king, 5 years, from 2019 to 2024."
        "霍...啦啦啦超过90%的人<le5>...?!9204"
    )
    tokenizer = EmiliaTokenizer()
    tokens = tokenizer.texts_to_tokens([text])
    print(f"tokens: {'|'.join(tokens[0])}")