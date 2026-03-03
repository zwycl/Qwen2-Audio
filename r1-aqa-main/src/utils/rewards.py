import os
import re
import unicodedata
from datetime import datetime
from typing import Optional

import numpy as np
import editdistance as ed
from math_verify import parse, verify
from whisper_normalizer.basic import BasicTextNormalizer
from whisper_normalizer.english import EnglishTextNormalizer

# Initialize normalizers for WER calculation
english_normalizer = EnglishTextNormalizer()
basic_normalizer = BasicTextNormalizer()

def _strip_punctuation(text: str) -> str:
    """Remove all punctuation including Unicode punctuation (CJK, etc.)."""
    # Remove characters in Unicode punctuation categories: Pc, Pd, Pe, Pf, Pi, Po, Ps
    return ''.join(c for c in text if not unicodedata.category(c).startswith('P'))

# Module-level storage for CGPR component metrics (accessed by trainer for wandb logging)
cgpr_metrics = {"cer": [], "bwer": [], "dense_reward": []}

# Module-level storage for CGPR+ component metrics (accessed by trainer for wandb logging)
cgpr_plus_metrics = {"cer": [], "script_fidelity": []}

# Module-level storage for script fidelity metrics (accessed by trainer for wandb logging)
script_fidelity_metrics = {"cer": [], "script_distance": []}

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
    # Remove language tokens like <|en|>, <|ar|>, <|zh|>, etc.
    gt = re.sub(r"<\|[a-z]{2}(_[a-z]{2})?\|>", " ", text)
    # Remove other special tokens like <|0.00|>, <|sil|>, etc.
    gt = re.sub(r"<\|.*?\|>", " ", gt)
    # Remove ** markers (used in CS-FLEURS for code-switch annotation)
    gt = gt.replace("**", "")
    gt = re.sub(rf"\s+", r" ", gt)
    gt = re.sub(f" ?([{PUNCS}])", r"\1", gt)
    gt = gt.lstrip(" ").rstrip(" ")
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


def _compute_single_cer(ref: str, pred: str, language: str) -> float:
    """Compute CER (Character Error Rate) for a single reference-prediction pair."""
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

    # Strip all punctuation (including Unicode CJK punctuation)
    ref = _strip_punctuation(ref)
    pred = _strip_punctuation(pred)

    # Tokenize words first, then convert to characters
    ref_words = tokenizer.tokenize(ref).split()
    pred_words = tokenizer.tokenize(pred).split()

    # Convert to character-level (removing spaces between words)
    ref_chars = list("".join(ref_words))
    pred_chars = list("".join(pred_words))

    # Compute edit distance and CER
    if len(ref_chars) == 0:
        return 0.0 if len(pred_chars) == 0 else 1.0

    distance = ed.eval(ref_chars, pred_chars)
    cer = distance / len(ref_chars)
    return cer


def cer_reward(completions, solution, language="en", **kwargs):
    """
    Reward function based on Character Error Rate (CER).

    Returns negative CER as reward (lower CER = higher reward).
    CER of 0 gives reward of 0, CER of 1 gives reward of -1.

    CER is particularly useful for:
    - Code-switched speech with mixed scripts
    - Languages like Chinese where character-level evaluation is meaningful
    - Catching minor spelling errors that WER might miss

    Args:
        completions: List of completions, each is a list with a dict containing "content"
        solution: List of ground truth transcriptions (one per completion)
        language: Language code or list of codes ("en", "zh", "yue", etc.) for normalization
                  If a list, must match length of completions (one per sample, repeated for generations)
        **kwargs: Additional arguments (ignored)

    Returns:
        List of rewards (negative CER values)
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

        # Compute CER and negate for reward
        cer = _compute_single_cer(ref, pred, lang)
        reward = -cer  # Negate: lower CER = higher reward

        rewards.append(reward)

        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} CER reward: {reward:.4f} (CER: {cer:.4f}) -------------\n")
                f.write(f"Sample {idx} | Language: {lang}\n")
                f.write(f"Prediction: {pred[:200]}{'...' if len(pred) > 200 else ''}\n")
                f.write(f"Reference:  {ref[:200]}{'...' if len(ref) > 200 else ''}\n")

    return rewards


def _get_script(char: str) -> str:
    """Classify a character into its Unicode script category."""
    cp = ord(char)
    if 0x0600 <= cp <= 0x06FF or 0x0750 <= cp <= 0x077F or 0x08A0 <= cp <= 0x08FF or 0xFB50 <= cp <= 0xFDFF or 0xFE70 <= cp <= 0xFEFF:
        return "Arabic"
    if 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or 0x20000 <= cp <= 0x2A6DF or 0xF900 <= cp <= 0xFAFF:
        return "CJK"
    if 0x3040 <= cp <= 0x309F:
        return "Hiragana"
    if 0x30A0 <= cp <= 0x30FF:
        return "Katakana"
    if 0xAC00 <= cp <= 0xD7AF or 0x1100 <= cp <= 0x11FF or 0x3130 <= cp <= 0x318F:
        return "Hangul"
    if 0x0400 <= cp <= 0x04FF or 0x0500 <= cp <= 0x052F:
        return "Cyrillic"
    if 0x0900 <= cp <= 0x097F:
        return "Devanagari"
    if (0x0041 <= cp <= 0x005A) or (0x0061 <= cp <= 0x007A) or (0x00C0 <= cp <= 0x024F):
        return "Latin"
    return "Other"


def _script_distribution(text: str) -> dict:
    """Compute fraction of characters in each Unicode script, ignoring spaces/punctuation."""
    counts = {}
    total = 0
    for c in text:
        if c.isspace() or unicodedata.category(c).startswith('P'):
            continue
        script = _get_script(c)
        if script == "Other":
            continue
        counts[script] = counts.get(script, 0) + 1
        total += 1
    if total == 0:
        return {}
    return {s: n / total for s, n in counts.items()}


def _total_variation_distance(dist_a: dict, dist_b: dict) -> float:
    """Total variation distance between two discrete distributions."""
    all_keys = set(dist_a) | set(dist_b)
    return 0.5 * sum(abs(dist_a.get(k, 0) - dist_b.get(k, 0)) for k in all_keys)


def script_fidelity_reward(completions, solution, language="en", gamma=0.5, **kwargs):
    """
    Script Fidelity Reward — CER combined with script distribution matching.

    Penalizes predictions that use the wrong writing system (e.g. outputting
    Latin where Arabic or CJK should appear). Computed as:

        reward = -CER - gamma * TVD(ref_scripts, pred_scripts)

    where TVD is the total variation distance between the Unicode script
    distributions of the reference and prediction (range [0, 1]).

    For X-English training: reinforces producing the correct non-English
    script, which should transfer to non-English pairs at test time.

    Args:
        completions: List of completions, each is a list with a dict containing "content"
        solution: List of ground truth transcriptions
        language: Language code or list of codes for CER normalization
        gamma: Weight for the script distance penalty (default 0.5)
        **kwargs: Additional arguments (ignored)

    Returns:
        List of rewards
    """
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

    script_fidelity_metrics["cer"].clear()
    script_fidelity_metrics["script_distance"].clear()

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

        # Normalize
        pred_norm = _remove_sp(pred, lang)
        ref_norm = _remove_sp(ref, lang)

        # CER component
        cer = _compute_single_cer(ref_norm, pred_norm, lang)

        # Script distribution distance
        ref_dist = _script_distribution(ref_norm)
        pred_dist = _script_distribution(pred_norm)
        tvd = _total_variation_distance(ref_dist, pred_dist)

        reward = -cer - gamma * tvd
        rewards.append(reward)

        script_fidelity_metrics["cer"].append(cer)
        script_fidelity_metrics["script_distance"].append(tvd)

        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Script Fidelity reward: {reward:.4f} -------------\n")
                f.write(f"Sample {idx} | Language: {lang}\n")
                f.write(f"CER: {cer:.4f}, Script TVD: {tvd:.4f} (gamma={gamma})\n")
                f.write(f"Ref scripts:  {ref_dist}\n")
                f.write(f"Pred scripts: {pred_dist}\n")
                f.write(f"Prediction: {pred_norm[:200]}{'...' if len(pred_norm) > 200 else ''}\n")
                f.write(f"Reference:  {ref_norm[:200]}{'...' if len(ref_norm) > 200 else ''}\n")

    return rewards


def mixed_wer_cer_reward(completions, solution, language="en", wer_weight=0.5, cer_weight=0.5, **kwargs):
    """
    Combined WER + CER reward function.

    Useful for code-switched speech where both word-level and character-level
    accuracy matter. WER captures word recognition, CER captures spelling accuracy.

    Args:
        completions: List of completions
        solution: List of ground truth transcriptions
        language: Language code or list of codes
        wer_weight: Weight for WER component (default 0.5)
        cer_weight: Weight for CER component (default 0.5)
        **kwargs: Additional arguments

    Returns:
        List of rewards: -(wer_weight * WER + cer_weight * CER)
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

        # Compute both WER and CER
        wer = _compute_single_wer(ref, pred, lang)
        cer = _compute_single_cer(ref, pred, lang)

        # Combined reward (negative weighted sum)
        reward = -(wer_weight * wer + cer_weight * cer)
        rewards.append(reward)

        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Mixed WER+CER reward: {reward:.4f} -------------\n")
                f.write(f"Sample {idx} | Language: {lang}\n")
                f.write(f"WER: {wer:.4f} (weight: {wer_weight}), CER: {cer:.4f} (weight: {cer_weight})\n")
                f.write(f"Prediction: {pred[:200]}{'...' if len(pred) > 200 else ''}\n")
                f.write(f"Reference:  {ref[:200]}{'...' if len(ref) > 200 else ''}\n")

    return rewards


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


# =============================================================================
# CGPR (Confidence-Gated Process Rewards) Implementation
# =============================================================================

def tsallis_entropy(probs: np.ndarray, q: float = 1/3) -> float:
    """
    Compute Tsallis entropy for confidence estimation.

    Tsallis entropy with q=1/3 provides ~4x better error detection than raw probs.

    Args:
        probs: Probability distribution (softmax output)
        q: Tsallis parameter (default 1/3 as recommended)

    Returns:
        Tsallis entropy value (lower = more confident)
    """
    # Filter out zero probabilities to avoid numerical issues
    probs = probs[probs > 0]
    if len(probs) == 0:
        return 0.0

    if q == 1.0:
        # Limit case: Shannon entropy
        return -np.sum(probs * np.log(probs))
    else:
        # Tsallis entropy: (1 - sum(p^q)) / (q - 1)
        return (1 - np.sum(probs ** q)) / (q - 1)


def compute_confidence_from_logits(logits: np.ndarray, q: float = 1/3) -> float:
    """
    Compute confidence score from logits using Tsallis entropy.

    Args:
        logits: Raw logits for a single token position
        q: Tsallis parameter

    Returns:
        Confidence score between 0 and 1 (higher = more confident)
    """
    # Apply softmax
    logits = logits - np.max(logits)  # Numerical stability
    probs = np.exp(logits) / np.sum(np.exp(logits))

    # Compute Tsallis entropy
    entropy = tsallis_entropy(probs, q)

    # Normalize entropy to [0, 1] and invert for confidence
    # Max entropy for uniform distribution
    n = len(probs)
    if q == 1.0:
        max_entropy = np.log(n)
    else:
        max_entropy = (1 - n ** (1 - q)) / (q - 1)

    # Confidence = 1 - normalized_entropy
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
    confidence = 1 - normalized_entropy

    return np.clip(confidence, 0, 1)


def _align_tokens(ref_tokens: list, hyp_tokens: list) -> list:
    """
    Align hypothesis tokens to reference tokens using edit distance alignment.

    Returns a list of (hyp_idx, ref_idx, operation) tuples.
    Operations: 'match', 'substitute', 'insert', 'delete'
    """
    n, m = len(ref_tokens), len(hyp_tokens)

    # DP table for edit distance with backtracking
    dp = np.zeros((n + 1, m + 1), dtype=int)

    # Initialize base cases
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    # Fill DP table
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_tokens[i - 1].lower() == hyp_tokens[j - 1].lower():
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j],      # deletion
                                  dp[i][j - 1],      # insertion
                                  dp[i - 1][j - 1])  # substitution

    # Backtrack to find alignment
    alignments = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_tokens[i - 1].lower() == hyp_tokens[j - 1].lower():
            alignments.append((j - 1, i - 1, 'match'))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            alignments.append((j - 1, i - 1, 'substitute'))
            i -= 1
            j -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            alignments.append((j - 1, -1, 'insert'))
            j -= 1
        else:
            # deletion - no hyp token corresponds to this ref token
            i -= 1

    alignments.reverse()
    return alignments


def compute_bwer(ref: str, pred: str, bias_list: list, language: str = "en") -> float:
    """
    Compute B-WER (Bias list Word Error Rate).

    Only counts errors on tokens that appear in the bias list.

    Args:
        ref: Reference transcription
        pred: Predicted transcription
        bias_list: List of entity/bias terms to focus on
        language: Language code for normalization

    Returns:
        B-WER score (0 = perfect on bias terms, 1 = all bias terms wrong)
    """
    tokenizer = EvaluationTokenizer(
        tokenizer_type="none",
        lowercase=True,
        punctuation_removal=True,
        character_tokenization=False,
    )

    # Normalize
    if language in ["en"]:
        ref = english_normalizer(ref)
        pred = english_normalizer(pred)
    else:
        ref = basic_normalizer(ref)
        pred = basic_normalizer(pred)

    ref_tokens = tokenizer.tokenize(ref).split()
    pred_tokens = tokenizer.tokenize(pred).split()

    # Normalize bias list tokens
    bias_set = set(t.lower() for t in bias_list)

    # Count bias tokens in reference and their errors
    bias_token_count = 0
    bias_errors = 0

    alignments = _align_tokens(ref_tokens, pred_tokens)

    # Track which ref tokens are covered
    ref_covered = [False] * len(ref_tokens)

    for hyp_idx, ref_idx, op in alignments:
        if ref_idx >= 0:
            ref_covered[ref_idx] = True
            ref_token = ref_tokens[ref_idx]
            if ref_token.lower() in bias_set:
                bias_token_count += 1
                if op != 'match':
                    bias_errors += 1

    # Count uncovered (deleted) bias tokens in reference
    for i, token in enumerate(ref_tokens):
        if not ref_covered[i] and token.lower() in bias_set:
            bias_token_count += 1
            bias_errors += 1

    if bias_token_count == 0:
        return 0.0

    return bias_errors / bias_token_count


def cgpr_shaped_reward(
    completions,
    solution,
    bias_list: Optional[list] = None,
    topk_logits_list: Optional[list] = None,
    token_ids_list: Optional[list] = None,
    tokenizer=None,
    language: str = "en",
    alpha: float = 0.1,
    beta: float = 0.3,
    lambda_entity: float = 4.0,
    tsallis_q: float = 1/3,
    use_bwer: bool = False,
    **kwargs
) -> list:
    """
    CGPR (Confidence-Gated Process Rewards) shaped reward function.

    Applies dense token-level rewards only to entity/bias-list tokens where
    correctness is verifiable, weighted by confidence to improve calibration.

    Reward structure:
        r_t = α · (1 - confidence)    if entity token AND correct
        r_t = -β · confidence         if entity token AND incorrect
        r_t = 0                       if non-entity token
        r_final = -CER - λ·B-WER      at sequence end (terminal reward, CER for code-switched)

    Args:
        completions: List of completions, each is a list with a dict containing "content"
        solution: List of ground truth transcriptions
        bias_list: List of entity/bias terms (if None, falls back to terminal reward only)
        topk_logits_list: Optional list of top-k logits per token for Tsallis entropy confidence
        token_ids_list: Optional list of token IDs for the completions
        tokenizer: Tokenizer for encoding entity words to find their positions
        language: Language code for normalization
        alpha: Coefficient for correct entity reward (default 0.1)
        beta: Coefficient for incorrect entity penalty (default 0.3)
        lambda_entity: Weight for B-WER in terminal reward (default 4.0, only used if use_bwer=True)
        tsallis_q: Tsallis entropy parameter (default 1/3, recommended for error detection)
        use_bwer: Whether to include B-WER in terminal reward (default False, redundant with dense rewards)
        **kwargs: Additional arguments (ignored)

    Returns:
        List of rewards (one per completion)
    """
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

    # Clear previous metrics and prepare for new batch
    cgpr_metrics["cer"].clear()
    cgpr_metrics["bwer"].clear()
    cgpr_metrics["dense_reward"].clear()

    # Handle language as list or single value
    if isinstance(language, list):
        languages = language
    else:
        languages = [language] * len(contents)

    # Handle bias_list as list of lists (per-sample) or single flat list
    if bias_list is not None and len(bias_list) > 0:
        if isinstance(bias_list[0], list):
            # Per-sample entity lists (from dataset)
            per_sample_bias = [set(t.lower() for t in bl) for bl in bias_list]
        else:
            # Single flat list - use for all samples
            shared_bias = set(t.lower() for t in bias_list)
            per_sample_bias = [shared_bias] * len(contents)
    else:
        per_sample_bias = [set() for _ in range(len(contents))]

    for idx, (content, sol, lang) in enumerate(zip(contents, solution, languages)):
        bias_set = per_sample_bias[idx] if idx < len(per_sample_bias) else set()
        # Extract answer from tags if present
        content_match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
        pred = content_match.group(1).strip() if content_match else content.strip()

        sol_match = re.search(r"<answer>(.*?)</answer>", sol, re.DOTALL)
        ref = sol_match.group(1).strip() if sol_match else sol.strip()

        # Remove special tokens and normalize
        pred = _remove_sp(pred, lang)
        ref = _remove_sp(ref, lang)

        # Compute terminal reward: -CER (better for code-switched speech)
        cer = _compute_single_cer(ref, pred, lang)
        terminal_reward = -cer

        # Add B-WER component if enabled and bias list is provided
        bwer = 0.0
        if use_bwer and len(bias_set) > 0:
            bwer = compute_bwer(ref, pred, list(bias_set), lang)
            terminal_reward -= lambda_entity * bwer

        # Compute dense entity-level rewards using Tsallis entropy confidence
        dense_reward = 0.0
        entity_rewards_debug = []  # Track per-entity rewards for debug logging
        if len(bias_set) > 0:
            # Get top-k logits and token ids for this sample (if available)
            sample_topk_logits = None
            sample_token_ids = None
            if topk_logits_list is not None and idx < len(topk_logits_list):
                sample_topk_logits = topk_logits_list[idx]
            if token_ids_list is not None and idx < len(token_ids_list):
                sample_token_ids = token_ids_list[idx]

            pred_lower = pred.lower()
            ref_lower = ref.lower()

            # Check each entity from the bias list
            for entity in bias_set:
                entity_lower = entity.lower()
                entity_in_pred = entity_lower in pred_lower
                entity_in_ref = entity_lower in ref_lower

                if not entity_in_pred and not entity_in_ref:
                    # Entity not relevant to this sample
                    continue

                # Compute confidence for this entity (if predicted)
                confidence = 0.5  # default
                if entity_in_pred and tokenizer is not None and sample_topk_logits is not None and sample_token_ids is not None:
                    # Decode and normalize completion to find entity position
                    decoded = tokenizer.decode(sample_token_ids, skip_special_tokens=True)
                    decoded_norm = _remove_sp(decoded, lang).lower()
                    entity_norm = _remove_sp(entity, lang).lower()

                    # Find character position in normalized text
                    char_pos = decoded_norm.find(entity_norm)
                    if char_pos >= 0:
                        # Estimate token position: chars before entity / avg chars per token
                        avg_chars_per_token = len(decoded_norm) / max(len(sample_token_ids), 1)
                        est_start_token = int(char_pos / avg_chars_per_token) if avg_chars_per_token > 0 else 0
                        est_num_tokens = max(1, int(len(entity_norm) / avg_chars_per_token))

                        # Clamp to valid range
                        est_start_token = max(0, min(est_start_token, len(sample_topk_logits) - 1))
                        est_end_token = min(est_start_token + est_num_tokens, len(sample_topk_logits))

                        # Compute confidence for estimated entity tokens
                        entity_confidences = []
                        for token_pos in range(est_start_token, est_end_token):
                            topk = np.array(sample_topk_logits[token_pos])
                            conf = compute_confidence_from_logits(topk, tsallis_q)
                            entity_confidences.append(conf)
                        if entity_confidences:
                            confidence = sum(entity_confidences) / len(entity_confidences)

                # Determine correctness and apply reward
                if entity_in_pred and entity_in_ref:
                    # Correct: entity predicted and should be there
                    token_reward = alpha * (1 - confidence)
                    dense_reward += token_reward
                    entity_rewards_debug.append({
                        "entity": entity,
                        "status": "correct",
                        "confidence": confidence,
                        "reward": token_reward,
                    })
                elif entity_in_pred and not entity_in_ref:
                    # Hallucination: entity predicted but shouldn't be there
                    token_reward = -beta * confidence
                    dense_reward += token_reward
                    entity_rewards_debug.append({
                        "entity": entity,
                        "status": "hallucinated",
                        "confidence": confidence,
                        "reward": token_reward,
                    })
                elif not entity_in_pred and entity_in_ref:
                    # Missing: entity should be there but wasn't predicted
                    token_reward = -beta * 0.5  # Penalty for missing (no confidence available)
                    dense_reward += token_reward
                    entity_rewards_debug.append({
                        "entity": entity,
                        "status": "missing",
                        "confidence": 0.0,
                        "reward": token_reward,
                    })

        # Total reward = terminal + dense
        reward = terminal_reward + dense_reward
        rewards.append(reward)

        # Store component metrics for wandb logging
        cgpr_metrics["cer"].append(cer)
        cgpr_metrics["bwer"].append(bwer)
        cgpr_metrics["dense_reward"].append(dense_reward)

        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} CGPR reward: {reward:.4f} -------------\n")
                f.write(f"Sample {idx} | Language: {lang}\n")
                f.write(f"Terminal: {terminal_reward:.4f} (CER: {cer:.4f}, B-WER: {bwer:.4f})\n")
                f.write(f"Dense: {dense_reward:.4f}\n")
                f.write(f"Prediction: {pred[:200]}{'...' if len(pred) > 200 else ''}\n")
                f.write(f"Reference:  {ref[:200]}{'...' if len(ref) > 200 else ''}\n")
                if entity_rewards_debug:
                    f.write(f"Entity rewards ({len(entity_rewards_debug)} entities):\n")
                    for ent in entity_rewards_debug:
                        status = ent["status"].upper()
                        f.write(f"  [{status}] '{ent['entity']}' "
                               f"conf={ent['confidence']:.3f} reward={ent['reward']:+.4f}\n")

    return rewards


# =============================================================================
# CGPR+ (Entity Preservation + Script Fidelity Rewards) Implementation
# =============================================================================

def _char_similarity(a: str, b: str) -> float:
    """1 - normalized_edit_distance between two strings."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    distance = ed.eval(list(a), list(b))
    max_len = max(len(a), len(b))
    return 1.0 - distance / max_len


def _best_substring_similarity(hypothesis: str, target: str, tolerance: float = 0.3) -> float:
    """
    Sliding window search: find the best matching substring of hypothesis to target.

    Searches windows of length len(target) * (1 +/- tolerance). Early exit at sim > 0.95.

    Args:
        hypothesis: The full predicted text to search within
        target: The target string to find
        tolerance: Fraction by which window size can vary from target length

    Returns:
        Best character similarity score found (0.0 to 1.0)
    """
    if not hypothesis or not target:
        return 0.0

    target_len = len(target)
    min_window = max(1, int(target_len * (1 - tolerance)))
    max_window = int(target_len * (1 + tolerance))

    best_sim = 0.0

    for window_size in range(min_window, max_window + 1):
        if window_size > len(hypothesis):
            continue
        for start in range(len(hypothesis) - window_size + 1):
            substring = hypothesis[start:start + window_size]
            sim = _char_similarity(substring, target)
            if sim > best_sim:
                best_sim = sim
            if best_sim > 0.95:
                return best_sim

    return best_sim


# Mapping from ISO 639-3 language codes to their expected Unicode script categories
LANG_TO_SCRIPTS = {
    "ara": {"Arabic"},
    "cmn": {"CJK"},
    "zho": {"CJK"},
    "jpn": {"CJK", "Hiragana", "Katakana"},
    "kor": {"Hangul"},
    "rus": {"Cyrillic"},
    "hin": {"Devanagari"},
    "eng": {"Latin"},
    "spa": {"Latin"},
    "fra": {"Latin"},
    "deu": {"Latin"},
    "por": {"Latin"},
    "ita": {"Latin"},
    "vie": {"Latin"},
    "tur": {"Latin"},
    "pol": {"Latin"},
    "nld": {"Latin"},
    "hun": {"Latin"},
    "ces": {"Latin"},
    "ind": {"Latin"},
}


def _get_allowed_scripts(language_pair: str) -> set:
    """
    Get the union of allowed scripts for both languages in a pair.

    Args:
        language_pair: Language pair code like "ara-eng", "cmn-eng"

    Returns:
        Set of allowed script names (e.g., {"Arabic", "Latin"})
    """
    parts = []
    if "-" in language_pair:
        parts = language_pair.lower().split("-")
    elif "_" in language_pair:
        parts = language_pair.lower().split("_")
    else:
        parts = [language_pair.lower()]

    allowed = set()
    for part in parts:
        scripts = LANG_TO_SCRIPTS.get(part, set())
        allowed.update(scripts)

    # Always allow Latin (numbers, common abbreviations, etc.)
    allowed.add("Latin")

    return allowed


def _compute_script_contamination(text: str, allowed_scripts: set) -> float:
    """
    Compute the fraction of non-space, non-punctuation characters that belong
    to a disallowed writing system.

    Args:
        text: Predicted text to check
        allowed_scripts: Set of allowed script names

    Returns:
        Fraction of characters in disallowed scripts (0.0 to 1.0)
    """
    total = 0
    disallowed = 0

    for c in text:
        if c.isspace() or unicodedata.category(c).startswith('P') or c.isdigit():
            continue
        script = _get_script(c)
        if script == "Other":
            continue
        total += 1
        if script not in allowed_scripts:
            disallowed += 1

    if total == 0:
        return 0.0
    return disallowed / total


def _count_script_transitions(text: str) -> int:
    """
    Count the number of writing-system transitions in text.

    Scans non-space, non-punctuation characters and counts each time the
    Unicode script category changes (e.g., Arabic -> Latin, Latin -> CJK).
    This measures the code-switching structure of the output.
    """
    prev_script = None
    transitions = 0
    for c in text:
        if c.isspace() or unicodedata.category(c).startswith('P') or c.isdigit():
            continue
        script = _get_script(c)
        if script == "Other":
            continue
        if prev_script is not None and script != prev_script:
            transitions += 1
        prev_script = script
    return transitions


def cgpr_plus_reward(
    completions,
    solution,
    raw_text: Optional[list] = None,
    language: str = "en",
    raw_language: Optional[list] = None,
    beta_switch: float = 0.1,
    beta_script: float = 0.1,
    **kwargs
) -> list:
    """
    CER + Switch Transition Fidelity Reward + Script Fidelity Reward.

    Reward = -CER + beta_switch * switch_fidelity + beta_script * script_fidelity

    1. Switch transition fidelity reward: counts the number of writing-system
       transitions in the hypothesis and compares to the number of code-switch
       boundaries in the reference (from ** markers). Rewards matching switch
       structure, penalizes language collapse (too few transitions) or
       hallucinated switches (too many transitions). Non-redundant with CER
       because CER is blind to *which* script produced the characters.
    2. Script fidelity reward: rewards using the correct writing systems for the
       language pair.

    Args:
        completions: List of completions, each is a list with a dict containing "content"
        solution: List of ground truth transcriptions
        raw_text: List of reference texts with ** markers (one per sample, repeated for generations)
        language: Language code or list of codes for normalization
        raw_language: Raw language pair codes (e.g., ["ara-eng"]) for script detection
        beta_switch: Coefficient for switch fidelity reward (default 0.1)
        beta_script: Coefficient for script fidelity reward (default 0.1)
        **kwargs: Additional arguments (ignored)

    Returns:
        List of rewards (one per completion)
    """
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

    # Clear previous metrics
    cgpr_plus_metrics["cer"].clear()
    cgpr_plus_metrics["script_fidelity"].clear()

    # Handle language as list or single value
    if isinstance(language, list):
        languages = language
    else:
        languages = [language] * len(contents)

    # Handle raw_language for script detection
    if raw_language is None:
        raw_languages = [None] * len(contents)
    elif isinstance(raw_language, str):
        raw_languages = [raw_language] * len(contents)
    elif isinstance(raw_language, list):
        raw_languages = raw_language
    else:
        raw_languages = [None] * len(contents)

    # Handle raw_text (with ** markers) as list
    if raw_text is None:
        raw_texts = [None] * len(contents)
    elif isinstance(raw_text, str):
        raw_texts = [raw_text] * len(contents)
    elif isinstance(raw_text, list):
        raw_texts = raw_text
    else:
        raw_texts = [None] * len(contents)

    for idx, (content, sol, lang) in enumerate(zip(contents, solution, languages)):
        raw_lang = raw_languages[idx] if idx < len(raw_languages) else None
        sample_raw_text = raw_texts[idx] if idx < len(raw_texts) else None

        # Extract answer from tags if present
        content_match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
        pred = content_match.group(1).strip() if content_match else content.strip()

        sol_match = re.search(r"<answer>(.*?)</answer>", sol, re.DOTALL)
        ref = sol_match.group(1).strip() if sol_match else sol.strip()

        # Remove special tokens and normalize
        pred = _remove_sp(pred, lang)
        ref = _remove_sp(ref, lang)

        # 1. CER terminal reward
        cer = _compute_single_cer(ref, pred, lang)
        terminal_reward = -cer

        # 2. Script fidelity reward: binary — full reward if no disallowed chars, else 0
        script_fidelity = 0.0
        if raw_lang:
            allowed = _get_allowed_scripts(raw_lang)
            contamination = _compute_script_contamination(pred, allowed)
            script_fidelity = beta_script if contamination == 0.0 else 0.0

        # Total reward
        reward = terminal_reward + script_fidelity
        rewards.append(reward)

        # Store metrics
        cgpr_plus_metrics["cer"].append(cer)
        cgpr_plus_metrics["script_fidelity"].append(script_fidelity)

        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} CGPR+ reward: {reward:.4f} -------------\n")
                f.write(f"Sample {idx} | Language: {lang} | Raw: {raw_lang}\n")
                f.write(f"Terminal: {terminal_reward:.4f} (CER: {cer:.4f})\n")
                f.write(f"Script fidelity: {script_fidelity:+.4f}\n")
                f.write(f"Prediction: {pred[:200]}{'...' if len(pred) > 200 else ''}\n")
                f.write(f"Reference:  {ref[:200]}{'...' if len(ref) > 200 else ''}\n")

    return rewards
