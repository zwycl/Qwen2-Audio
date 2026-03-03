import warnings
# Suppress NNPACK warnings (harmless - just means CPU optimizations unavailable)
warnings.filterwarnings("ignore", message=".*NNPACK.*")

from .dataset import AudioDataset
from .afrispeech_dataset import AfriSpeechASRDataset, AfriSpeechASRDatasetFromHF
from .switchlingua_dataset import SwitchLinguaDataset

# Two-step training: Refinement prompt templates (used for second pass)
REFINEMENT_PROMPT_TEMPLATE = (
    "You are a speech transcription system. A previous transcription attempt produced: \"{draft_transcription}\". "
    "Listen to the audio again and correct any errors. "
    "The following names/terms may appear: {entity_str}. Use correct spelling for these terms. "
    "Output the corrected transcription in <answer> </answer>."
)

REFINEMENT_NO_CONTEXT_PROMPT_TEMPLATE = (
    "You are a speech transcription system. A previous transcription attempt produced: \"{draft_transcription}\". "
    "Listen to the audio again and correct any errors. "
    "Output the corrected transcription in <answer> </answer>."
)

__all__ = [
    "AudioDataset",
    "AfriSpeechASRDataset",
    "AfriSpeechASRDatasetFromHF",
    "SwitchLinguaDataset",
    "REFINEMENT_PROMPT_TEMPLATE",
    "REFINEMENT_NO_CONTEXT_PROMPT_TEMPLATE",
]
