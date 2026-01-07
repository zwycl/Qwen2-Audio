import os
import re
import unicodedata
from datetime import datetime

import editdistance as ed
from math_verify import parse, verify
from whisper_normalizer.basic import BasicTextNormalizer
from whisper_normalizer.english import EnglishTextNormalizer

# Initialize normalizers for WER calculation
english_normalizer = EnglishTextNormalizer()
basic_normalizer = BasicTextNormalizer()

# Optional Chinese support - not needed for AfriSpeech (English)
try:
    import zhconv
    from cn_tn import TextNorm
    chinese_normalizer = TextNorm(
        to_banjiao=False,
        to_upper=False,
        to_lower=False,
        remove_fillers=False,
        remove_erhua=False,
        check_chars=False,
        remove_space=False,
        cc_mode='',
    )
    HAS_CHINESE_SUPPORT = True
except ImportError:
    zhconv = None
    chinese_normalizer = None
    HAS_CHINESE_SUPPORT = False


class EvaluationTokenizer:
    """A tokenizer for WER evaluation with punctuation removal and lowercasing."""

    SPACE = chr(32)
    SPACE_ESCAPE = chr(9601)

    def __init__(
        self,
        tokenizer_type: str = "13a",
        lowercase: bool = False,
        punctuation_removal: bool = False,
        character_tokenization: bool = False,
    ):
        from sacrebleu.tokenizers import TOKENIZERS
        assert tokenizer_type in TOKENIZERS, f"{tokenizer_type}, {TOKENIZERS}"
        self.lowercase = lowercase
        self.punctuation_removal = punctuation_removal
        self.character_tokenization = character_tokenization
        self.tokenizer = TOKENIZERS[tokenizer_type]

    @classmethod
    def remove_punctuation(cls, sent: str):
        """Remove punctuation based on Unicode category."""
        return cls.SPACE.join(
            t for t in sent.split(cls.SPACE) if not all(unicodedata.category(c)[0] == "P" for c in t)
        )

    def tokenize(self, sent: str):
        tokenized = self.tokenizer()(sent)
        if self.punctuation_removal:
            tokenized = self.remove_punctuation(tokenized)
        if self.character_tokenization:
            tokenized = self.SPACE.join(list(tokenized.replace(self.SPACE, self.SPACE_ESCAPE)))
        if self.lowercase:
            tokenized = tokenized.lower()
        return tokenized


def accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol in zip(contents, solution):
        reward = 0.0
        # Try symbolic verification first
        try:
            answer = parse(content)
            if float(verify(answer, parse(sol))) > 0:
                reward = 1.0
        except Exception:
            pass  # Continue to next verification method if this fails

        # If symbolic verification failed, try string matching
        if reward == 0.0:
            try:
                # Extract answer from solution if it has think/answer tags
                sol_match = re.search(r"<answer>(.*?)</answer>", sol)
                ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

                # Extract answer from content if it has think/answer tags
                content_match = re.search(r"<answer>(.*?)</answer>", content)
                student_answer = content_match.group(1).strip() if content_match else content.strip()

                # Compare the extracted answers
                if student_answer == ground_truth:
                    reward = 1.0
            except Exception:
                pass  # Keep reward as 0.0 if both methods fail

        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<answer>.*?</answer>"
    # If you want to improve the thinking process, uncomment the next line and design your strategy.
    # pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


PUNCS = '!,.?;:'


def _remove_sp(text, language):
    """Remove special tokens and normalize spacing."""
    gt = re.sub(r"<\|.*?\|>", " ", text)
    gt = re.sub(rf"\s+", r" ", gt)
    gt = re.sub(f" ?([{PUNCS}])", r"\1", gt)
    gt = gt.lstrip(" ")
    if language == "zh":
        gt = re.sub(rf"\s+", r"", gt)
    return gt


def _compute_single_wer(ref: str, pred: str, language: str) -> float:
    """Compute WER for a single reference-prediction pair."""
    tokenizer = EvaluationTokenizer(
        tokenizer_type="none",
        lowercase=True,
        punctuation_removal=True,
        character_tokenization=False,
    )

    # Apply language-specific normalization
    if language in ["yue"] and HAS_CHINESE_SUPPORT:
        ref = zhconv.convert(ref, 'zh-cn')
        pred = zhconv.convert(pred, 'zh-cn')
        ref = basic_normalizer(ref)
        pred = basic_normalizer(pred)
    elif language in ["en"]:
        ref = english_normalizer(ref)
        pred = english_normalizer(pred)
    elif language in ["zh"] and HAS_CHINESE_SUPPORT:
        ref = chinese_normalizer(ref)
        pred = chinese_normalizer(pred)
    else:
        ref = basic_normalizer(ref)
        pred = basic_normalizer(pred)

    # Tokenize
    ref_items = tokenizer.tokenize(ref).split()
    pred_items = tokenizer.tokenize(pred).split()

    # For Chinese/Cantonese, use character-level tokenization
    if language in ["zh", "yue"] and HAS_CHINESE_SUPPORT:
        ref_items = [x for x in "".join(ref_items)]
        pred_items = [x for x in "".join(pred_items)]

    # Compute edit distance and WER
    if len(ref_items) == 0:
        return 0.0 if len(pred_items) == 0 else 1.0

    distance = ed.eval(ref_items, pred_items)
    wer = distance / len(ref_items)
    return wer


def wer_reward(completions, solution, language="en", **kwargs):
    """
    Reward function based on Word Error Rate (WER).

    Returns negative WER as reward (lower WER = higher reward).
    WER of 0 gives reward of 0, WER of 1 gives reward of -1.

    Args:
        completions: List of completions, each is a list with a dict containing "content"
        solution: List of ground truth transcriptions (one per completion)
        language: Language code or list of codes ("en", "zh", "yue", etc.) for normalization
                  If a list, must match length of completions (one per sample, repeated for generations)
        **kwargs: Additional arguments (ignored)

    Returns:
        List of rewards (negative WER values)
    """
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

    # Handle language as list (one per completion) or single value
    if isinstance(language, list):
        languages = language
    else:
        languages = [language] * len(contents)

    for idx, (content, sol, lang) in enumerate(zip(contents, solution, languages)):
        # Extract answer from tags if present
        content_match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
        pred = content_match.group(1).strip() if content_match else content.strip()

        sol_match = re.search(r"<answer>(.*?)</answer>", sol, re.DOTALL)
        ref = sol_match.group(1).strip() if sol_match else sol.strip()

        # Remove special tokens and normalize
        pred = _remove_sp(pred, lang)
        ref = _remove_sp(ref, lang)

        # Compute WER and negate for reward
        wer = _compute_single_wer(ref, pred, lang)
        reward = -wer  # Negate: lower WER = higher reward

        rewards.append(reward)

        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} WER reward: {reward:.4f} (WER: {wer:.4f}) -------------\n")
                f.write(f"Sample {idx} | Language: {lang}\n")
                f.write(f"Prediction: {pred[:200]}{'...' if len(pred) > 200 else ''}\n")
                f.write(f"Reference:  {ref[:200]}{'...' if len(ref) > 200 else ''}\n")

    return rewards
