# Copyright 2025 The HuggingFace Team. All rights reserved.
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
import os
import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Sized, Union

import torch
import torch.utils.data
import transformers
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    Qwen2AudioForConditionalGeneration,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url, selective_log_softmax
from trl.trainer.callbacks import SyncRefModelCallback

from utils.rewards import cgpr_metrics, cgpr_plus_metrics
from dataset import REFINEMENT_PROMPT_TEMPLATE, REFINEMENT_NO_CONTEXT_PROMPT_TEMPLATE

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


# Based on R1-V code base, https://github.com/Deep-Agent/R1-V/blob/main/src/r1-v/src/open_r1/trainer/grpo_trainer.py
class GRPOTrainer(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs="weqweasdas/RM-Gemma-2B",
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        attn_implementation: str = "flash_attention_2",
        two_step_training: bool = False,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            if "Qwen2-Audio" in model_id:
                model = Qwen2AudioForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            else:
                model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        # Freeze audio encoder (as per GRPO for ASR paper)
        if hasattr(model, 'audio_tower'):
            for param in model.audio_tower.parameters():
                param.requires_grad = False
            print("Frozen audio_tower parameters")

        # Reference model
        if is_deepspeed_zero3_enabled():
            if "Qwen2-Audio" in model_id:
                 self.ref_model = Qwen2AudioForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            else:
                self.ref_model = AutoModelForCausalLM.from_pretrained(model_id, **model_init_kwargs)
        elif not is_peft_model(model):
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)
        else:
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None

        # Processing class
        if processing_class is None:
            if "Qwen2-Audio" in model_id:
                processing_class = AutoProcessor.from_pretrained(model_id)
                processing_class.pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
            else:
                processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.two_step_training = two_step_training

        # Two-step training: cache for gradient accumulation
        if two_step_training:
            self._two_step_cache = []  # List of (inputs, completions) tuples
            self._accumulation_counter = 0

        self.beta = args.beta

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        set_seed(args.seed, device_specific=True)

        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            temperature=args.temperature,
            num_return_sequences=self.num_generations,
            pad_token_id=processing_class.pad_token_id,
            eos_token_id=processing_class.eos_token_id,
            renormalize_logits=True,  # Prevent NaN/Inf in sampling
        )

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        if args.sync_ref_model:
            self.add_callback(SyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator))

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]
    
    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(self, model, input_ids, attention_mask, features_values, features_masks, return_topk=False, topk=10):
        logits = model(input_ids, attention_mask=attention_mask, input_features=features_values, feature_attention_mask=features_masks).logits  # (B, L, V)
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        per_token_logps = selective_log_softmax(logits, input_ids)

        if return_topk:
            # Extract top-k logits for entropy-based confidence computation
            topk_logits, _ = torch.topk(logits, k=topk, dim=-1)  # (B, L-1, k)
            return per_token_logps, topk_logits
        return per_token_logps


    # Trainer "prepares" the inputs before calling `compute_loss`. It converts to tensor and move to device.
    # Since we preprocess the data in `compute_loss`, we need to override this method to skip this step.
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        return inputs

    def _extract_draft_transcription(self, completion_text: str) -> str:
        """Extract the transcription from completion text, handling <answer> tags."""
        import re
        # Try to extract content from <answer> tags
        match = re.search(r"<answer>(.*?)</answer>", completion_text, re.DOTALL)
        if match:
            draft = match.group(1).strip()
        else:
            # Fallback: use the whole completion
            draft = completion_text.strip()

        # Defensive: ensure non-empty and reasonable length
        if not draft:
            draft = "[empty transcription]"
        # Truncate very long drafts to avoid tokenization issues
        if len(draft) > 2000:
            draft = draft[:2000]
        return draft

    def _build_refinement_prompts(self, inputs: list, best_completions: list) -> list:
        """
        Build refinement prompts for the second pass of two-step training.

        Args:
            inputs: Original input samples (batch)
            best_completions: Best draft completion per sample (one per sample)

        Returns:
            List of refinement prompts, one per sample
        """
        refinement_prompts = []

        for i, inp in enumerate(inputs):
            entity_list = inp.get("entity_list", [])
            audio_path = ""
            # Extract audio_url from original prompt
            for msg in inp.get("prompt", []):
                if msg.get("role") == "user":
                    for content_item in msg.get("content", []):
                        if content_item.get("type") == "audio":
                            audio_path = content_item.get("audio_url", "")
                            break

            # Get best draft for this sample
            draft_completion = best_completions[i]

            # Extract draft text from completion
            if isinstance(draft_completion, list):
                draft_text = draft_completion[0]["content"]
            else:
                draft_text = draft_completion
            draft_transcription = self._extract_draft_transcription(draft_text)

            # Build refinement prompt text
            if entity_list:
                entity_str = ", ".join(entity_list)
                prompt_text = REFINEMENT_PROMPT_TEMPLATE.format(
                    draft_transcription=draft_transcription,
                    entity_str=entity_str
                )
            else:
                prompt_text = REFINEMENT_NO_CONTEXT_PROMPT_TEMPLATE.format(
                    draft_transcription=draft_transcription
                )

            # Build prompt in the same format as original
            refinement_prompt = {
                "prompt": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "audio", "audio_url": audio_path},
                            {"type": "text", "text": prompt_text},
                        ],
                    }
                ],
                "audio": inp.get("audio"),
                "solution": inp.get("solution"),
                "entity_list": entity_list,
                # Preserve other metadata
                "language": inp.get("language"),
                "dataset_name": inp.get("dataset_name"),
                "uniq_id": inp.get("uniq_id"),
                "chunk_start": inp.get("chunk_start"),
                "chunk_end": inp.get("chunk_end"),
            }
            refinement_prompts.append(refinement_prompt)

        return refinement_prompts

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        try:
            loss = self._compute_loss_inner(model, inputs, num_items_in_batch)
        except Exception as e:
            logging.warning(f"Compute loss failed, skipping batch: {e}")
            # Return zero loss to skip this batch
            device = self.accelerator.device
            loss = torch.tensor(0.0, device=device, requires_grad=True)

        # Ensure loss is a scalar tensor (required by DeepSpeed)
        if not isinstance(loss, torch.Tensor):
            device = self.accelerator.device
            loss = torch.tensor(float(loss), device=device, requires_grad=True)
        if loss.dim() != 0:
            loss = loss.mean()  # Force to scalar
        if not loss.requires_grad:
            # Wrap in a computation that requires grad
            zero = torch.zeros(1, device=loss.device, requires_grad=True).sum()
            loss = loss + zero
        return loss

    def _compute_loss_inner(self, model, inputs, num_items_in_batch=None):
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        audios = [x["audio"] for x in inputs]
        prompt_inputs = self.processing_class(
            text=prompts_text,
            audio=audios,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)

        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        features_values = prompt_inputs["input_features"]
        features_masks = prompt_inputs["feature_attention_mask"]
        
        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        # Generate completions using the DeepSpeed module
        if hasattr(model, 'module'):
            unwrapped_model = model.module
        else:
            unwrapped_model = self.accelerator.unwrap_model(model)

        # Disable gradient checkpointing for generation if enabled
        is_gradient_checkpointing = getattr(unwrapped_model, 'is_gradient_checkpointing', False)
        if is_gradient_checkpointing:
            unwrapped_model.gradient_checkpointing_disable()

        # Detect model device and move inputs accordingly
        model_device = next(unwrapped_model.parameters()).device
        generation_inputs = {k: v.to(model_device) if isinstance(v, torch.Tensor) else v for k, v in prompt_inputs.items()}

        # Check for NaN/Inf in input features before generation
        if torch.isnan(generation_inputs["input_features"]).any() or torch.isinf(generation_inputs["input_features"]).any():
            raise ValueError("NaN/Inf detected in input features, skipping batch")

        with torch.no_grad():
            prompt_completion_ids = unwrapped_model.generate(**generation_inputs, generation_config=self.generation_config)
            prompt_completion_ids = prompt_completion_ids.to(self.accelerator.device)

        # Re-enable gradient checkpointing if it was enabled
        if is_gradient_checkpointing:
            unwrapped_model.gradient_checkpointing_enable()

        # Use actual full prompt length from generation input, not truncated prompt_ids
        # (prompt_ids may be truncated by max_prompt_length, but generation uses full prompt_inputs)
        prompt_length = prompt_inputs["input_ids"].size(1)

        # Debug output
        if self.accelerator.is_main_process:
            print(f"DEBUG generation: prompt_ids.shape={prompt_ids.shape}, full_prompt_length={prompt_length}, prompt_completion_ids.shape={prompt_completion_ids.shape}")
            if prompt_completion_ids.size(1) > prompt_length:
                split_tokens = prompt_completion_ids[0, max(0, prompt_length-3):prompt_length+10].tolist()
                decoded_split = self.processing_class.decode(split_tokens)
                print(f"DEBUG generation: tokens around split point: {split_tokens}")
                print(f"DEBUG generation: decoded around split: {repr(decoded_split)}")

        prompt_ids = prompt_completion_ids[:, :prompt_length]
        completion_ids = prompt_completion_ids[:, prompt_length:]

        # Create prompt_mask matching full prompt length (original was truncated)
        prompt_mask = prompt_inputs["attention_mask"].repeat_interleave(self.num_generations, dim=0)

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        device = self.accelerator.device
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B*G, P+C)
        features_values = features_values.repeat(self.num_generations, 1, 1)
        features_masks = features_masks.repeat_interleave(self.num_generations, dim=0)

        per_token_logps, topk_logits = self._get_per_token_logps(model, prompt_completion_ids, attention_mask, features_values, features_masks, return_topk=True, topk=10)
        # Get rid of the prompt (-1 because of the shift done in get_per_token_logps)
        per_token_logps = per_token_logps[:, prompt_length - 1 :]
        topk_logits = topk_logits[:, prompt_length - 1 :]  # (B*G, completion_length, 10)

        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(self.ref_model, prompt_completion_ids, attention_mask, features_values, features_masks)
            else:
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(model, prompt_completion_ids, attention_mask, features_values, features_masks)
        ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1 :]

        # Compute the KL divergence between the model and the reference model
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

        # Decode the generated completions
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions]

        # Debug logging: show chunk info, ground truth, and generated transcription
        if self.accelerator.is_main_process and self.state.global_step % 2 == 0:
            import re
            print("\n" + "="*80)
            print(f"DEBUG: Step {self.state.global_step} - Chunk Transcriptions")
            print(f"  Batch size: {len(inputs)} chunks, {self.num_generations} generations each")
            print(f"  completion_ids shape: {completion_ids.shape}")
            print("="*80)

            for i, inp in enumerate(inputs):  # Show all chunks in batch
                # Chunk metadata
                chunk_id = inp.get("uniq_id", f"chunk_{i}")
                chunk_start = inp.get("chunk_start", 0)
                chunk_end = inp.get("chunk_end", 0)
                audio = inp.get("audio")
                audio_duration = len(audio) / 16000 if audio is not None else 0
                solution = inp.get("solution", "N/A")
                entity_list = inp.get("entity_list", [])

                # Extract prompt text
                prompt = inp.get("prompt", [])
                prompt_text = ""
                for msg in prompt:
                    if msg.get("role") == "user":
                        for content_item in msg.get("content", []):
                            if content_item.get("type") == "text":
                                prompt_text = content_item.get("text", "")
                                break

                # Extract ground truth from solution
                sol_match = re.search(r"<answer>(.*?)</answer>", str(solution), re.DOTALL)
                ref = sol_match.group(1).strip() if sol_match else str(solution).strip()

                print(f"\n[Chunk {i+1}/{len(inputs)}] ID: {chunk_id} ({chunk_start:.1f}s - {chunk_end:.1f}s)")
                print(f"  Entities: {entity_list}")
                print(f"  Prompt: {prompt_text}")
                print(f"  Ref: {ref}")

                # Show first generation only for this chunk
                gen_idx = i * self.num_generations
                comp = completions[gen_idx]
                content = comp[0]["content"] if isinstance(comp, list) else comp
                content_match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
                pred = content_match.group(1).strip() if content_match else content.strip()
                print(f"  Gen: {pred}")

            print("="*80 + "\n")

        # Compute the rewards
        prompts = [prompt for prompt in prompts for _ in range(self.num_generations)]

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
            else:
                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
                for key in reward_kwargs:
                    for example in inputs:
                        # Repeat each value in the column for `num_generations` times
                        reward_kwargs[key].extend([example[key]] * self.num_generations)

                # Pass top-k logits for CGPR dense reward Tsallis entropy confidence
                # topk_logits shape: (batch*G, completion_length, 10)
                reward_kwargs["topk_logits_list"] = topk_logits.cpu().tolist()
                reward_kwargs["token_ids_list"] = completion_ids.cpu().tolist()
                # Pass tokenizer for entity word tokenization
                reward_kwargs["tokenizer"] = self.processing_class.tokenizer if hasattr(self.processing_class, 'tokenizer') else self.processing_class

                output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # Sum the rewards from all reward functions
        rewards = rewards_per_func.sum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # Standard policy gradient: -log_prob * advantage + KL penalty
        per_token_loss = -(per_token_logps * advantages.unsqueeze(1)) + self.beta * per_token_kl
        # Avoid division by zero for empty completions
        completion_lengths = completion_mask.sum(dim=1).clamp(min=1)
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_lengths).mean()
        # Handle NaN/Inf by replacing with zero (keep requires_grad from original)
        if torch.isnan(loss) or torch.isinf(loss):
            loss = loss * 0.0  # Preserves requires_grad and device

        # Cache best completions for two-step training (accumulate across gradient accumulation steps)
        if self.two_step_training:
            # Pick best draft per sample based on reward
            best_completions = []
            for i in range(len(inputs)):
                start_idx = i * self.num_generations
                end_idx = start_idx + self.num_generations
                sample_rewards = rewards[start_idx:end_idx]
                best_idx = sample_rewards.argmax().item()
                best_completions.append(completions[start_idx + best_idx])
            self._two_step_cache.append((inputs, best_completions))

        # Log the metrics
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        # Log CGPR component metrics if available
        if len(cgpr_metrics["cer"]) > 0:
            self._metrics["cgpr/cer"].append(sum(cgpr_metrics["cer"]) / len(cgpr_metrics["cer"]))
            self._metrics["cgpr/bwer"].append(sum(cgpr_metrics["bwer"]) / len(cgpr_metrics["bwer"]))
            self._metrics["cgpr/dense_reward"].append(sum(cgpr_metrics["dense_reward"]) / len(cgpr_metrics["dense_reward"]))

        # Log CGPR+ component metrics if available
        if len(cgpr_plus_metrics["cer"]) > 0:
            self._metrics["cgpr_plus/cer"].append(sum(cgpr_plus_metrics["cer"]) / len(cgpr_plus_metrics["cer"]))
            self._metrics["cgpr_plus/script_fidelity"].append(sum(cgpr_plus_metrics["script_fidelity"]) / len(cgpr_plus_metrics["script_fidelity"]))

        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())
        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())

        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_lengths).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        return loss

    def compute_refinement_loss(self, model, inputs, completions):
        """
        Compute loss for the refinement pass (second pass of two-step training).

        Args:
            model: The model to compute loss for
            inputs: Original input samples
            completions: Draft completions from first pass
        """
        try:
            loss = self._compute_refinement_loss_inner(model, inputs, completions)
        except Exception as e:
            logging.warning(f"Refinement pass failed, skipping batch: {e}")
            # Return zero loss to skip this batch
            device = self.accelerator.device
            loss = torch.tensor(0.0, device=device, requires_grad=True)

        # Ensure loss is a scalar tensor (required by DeepSpeed)
        if not isinstance(loss, torch.Tensor):
            device = self.accelerator.device
            loss = torch.tensor(float(loss), device=device, requires_grad=True)
        if loss.dim() != 0:
            loss = loss.mean()  # Force to scalar
        if not loss.requires_grad:
            # Wrap in a computation that requires grad
            zero = torch.zeros(1, device=loss.device, requires_grad=True).sum()
            loss = loss + zero
        return loss

    def _compute_refinement_loss_inner(self, model, inputs, completions):
        """Inner implementation of refinement loss computation."""
        import re

        # Build refinement prompts using draft completions
        refinement_inputs = self._build_refinement_prompts(inputs, completions)

        # Process refinement prompts (one per draft, so batch*G items)
        ref_prompts = [x["prompt"] for x in refinement_inputs]
        ref_prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in refinement_inputs]
        ref_audios = [x["audio"] for x in refinement_inputs]

        ref_prompt_inputs = self.processing_class(
            text=ref_prompts_text,
            audio=ref_audios,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        ref_prompt_inputs = super()._prepare_inputs(ref_prompt_inputs)

        ref_features_values = ref_prompt_inputs["input_features"]
        ref_features_masks = ref_prompt_inputs["feature_attention_mask"]

        # Get model for generation
        if hasattr(model, 'module'):
            unwrapped_model = model.module
        else:
            unwrapped_model = self.accelerator.unwrap_model(model)

        is_gradient_checkpointing = getattr(unwrapped_model, 'is_gradient_checkpointing', False)
        model_device = next(unwrapped_model.parameters()).device
        device = self.accelerator.device

        # Generate refinement completions (num_generations per sample, using best draft)
        ref_generation_inputs = {k: v.to(model_device) if isinstance(v, torch.Tensor) else v for k, v in ref_prompt_inputs.items()}

        ref_generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            temperature=self.generation_config.temperature,
            num_return_sequences=self.num_generations,  # Generate G refinements per best draft
            pad_token_id=self.processing_class.pad_token_id,
            eos_token_id=self.processing_class.eos_token_id,
            renormalize_logits=True,  # Prevent NaN/Inf in sampling
        )

        if is_gradient_checkpointing:
            unwrapped_model.gradient_checkpointing_disable()

        # Check for NaN/Inf in input features before generation
        if torch.isnan(ref_generation_inputs["input_features"]).any() or torch.isinf(ref_generation_inputs["input_features"]).any():
            raise ValueError("NaN/Inf detected in input features, skipping batch")

        with torch.no_grad():
            ref_prompt_completion_ids = unwrapped_model.generate(**ref_generation_inputs, generation_config=ref_generation_config)
            ref_prompt_completion_ids = ref_prompt_completion_ids.to(device)

        if is_gradient_checkpointing:
            unwrapped_model.gradient_checkpointing_enable()

        ref_prompt_length = ref_prompt_inputs["input_ids"].size(1)
        ref_completion_ids = ref_prompt_completion_ids[:, ref_prompt_length:]

        # Mask everything after the first EOS token
        ref_is_eos = ref_completion_ids == self.processing_class.eos_token_id
        ref_eos_idx = torch.full((ref_is_eos.size(0),), ref_is_eos.size(1), dtype=torch.long, device=device)
        ref_eos_idx[ref_is_eos.any(dim=1)] = ref_is_eos.int().argmax(dim=1)[ref_is_eos.any(dim=1)]
        ref_sequence_indices = torch.arange(ref_is_eos.size(1), device=device).expand(ref_is_eos.size(0), -1)
        ref_completion_mask = (ref_sequence_indices <= ref_eos_idx.unsqueeze(1)).int()

        # Create attention mask (expand for num_generations)
        ref_prompt_mask = ref_prompt_inputs["attention_mask"].repeat_interleave(self.num_generations, dim=0)
        ref_attention_mask = torch.cat([ref_prompt_mask, ref_completion_mask], dim=1)

        # Expand features for num_generations
        ref_features_values = ref_features_values.repeat(self.num_generations, 1, 1)
        ref_features_masks = ref_features_masks.repeat_interleave(self.num_generations, dim=0)

        # Compute log probs for refinement pass
        ref_per_token_logps, ref_topk_logits = self._get_per_token_logps(
            model, ref_prompt_completion_ids, ref_attention_mask,
            ref_features_values, ref_features_masks, return_topk=True, topk=10
        )
        ref_per_token_logps = ref_per_token_logps[:, ref_prompt_length - 1 :]
        ref_topk_logits = ref_topk_logits[:, ref_prompt_length - 1 :]

        with torch.inference_mode():
            if self.ref_model is not None:
                ref_ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, ref_prompt_completion_ids, ref_attention_mask,
                    ref_features_values, ref_features_masks
                )
            else:
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_ref_per_token_logps = self._get_per_token_logps(
                        model, ref_prompt_completion_ids, ref_attention_mask,
                        ref_features_values, ref_features_masks
                    )
        ref_ref_per_token_logps = ref_ref_per_token_logps[:, ref_prompt_length - 1 :]

        # Compute KL
        ref_per_token_kl = torch.exp(ref_ref_per_token_logps - ref_per_token_logps) - (ref_ref_per_token_logps - ref_per_token_logps) - 1

        # Decode refinement completions
        ref_completions = self.processing_class.batch_decode(ref_completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            ref_completions = [[{"role": "assistant", "content": completion}] for completion in ref_completions]

        # Debug logging
        if self.accelerator.is_main_process and self.state.global_step % 2 == 0:
            print("\n" + "="*80)
            print(f"DEBUG: Step {self.state.global_step} - Refinement Pass (Pass 2)")
            print("="*80)
            for i in range(min(2, len(inputs))):
                # Best draft from pass 1 (one per sample)
                draft_comp = completions[i]
                draft_text = draft_comp[0]["content"] if isinstance(draft_comp, list) else draft_comp
                draft_match = re.search(r"<answer>(.*?)</answer>", draft_text, re.DOTALL)
                draft_pred = draft_match.group(1).strip() if draft_match else draft_text.strip()

                # First refined output from pass 2
                ref_idx = i * self.num_generations
                ref_comp = ref_completions[ref_idx]
                ref_text = ref_comp[0]["content"] if isinstance(ref_comp, list) else ref_comp
                ref_match = re.search(r"<answer>(.*?)</answer>", ref_text, re.DOTALL)
                ref_pred = ref_match.group(1).strip() if ref_match else ref_text.strip()

                # Ground truth
                solution = inputs[i].get("solution", "")
                sol_match = re.search(r"<answer>(.*?)</answer>", str(solution), re.DOTALL)
                ground_truth = sol_match.group(1).strip() if sol_match else str(solution).strip()

                print(f"\n[Sample {i+1}]")
                print(f"  Ground Truth: {ground_truth}")
                print(f"  Best Draft (Pass 1): {draft_pred}")
                print(f"  Refined (Pass 2): {ref_pred}")
            print("="*80 + "\n")

        # Compute rewards for refinement pass
        # Expand prompts to match completions (num_generations per sample)
        ref_prompts_expanded = [p for p in ref_prompts for _ in range(self.num_generations)]
        num_total_completions = len(inputs) * self.num_generations

        ref_rewards_per_func = torch.zeros(num_total_completions, len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(ref_prompts_expanded, ref_completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(ref_prompts_expanded, ref_completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    ref_rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]
            else:
                # Expand reward kwargs for num_generations
                reward_kwargs = {key: [] for key in refinement_inputs[0].keys() if key not in ["prompt", "completion"]}
                for key in reward_kwargs:
                    for example in refinement_inputs:
                        reward_kwargs[key].extend([example[key]] * self.num_generations)

                reward_kwargs["topk_logits_list"] = ref_topk_logits.cpu().tolist()
                reward_kwargs["token_ids_list"] = ref_completion_ids.cpu().tolist()
                reward_kwargs["tokenizer"] = self.processing_class.tokenizer if hasattr(self.processing_class, 'tokenizer') else self.processing_class

                output_reward_func = reward_func(prompts=ref_prompts_expanded, completions=ref_completions, **reward_kwargs)
                ref_rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        ref_rewards = ref_rewards_per_func.sum(dim=1)

        # Compute advantages (group by original sample, num_generations per sample)
        ref_mean_grouped_rewards = ref_rewards.view(-1, self.num_generations).mean(dim=1)
        ref_std_grouped_rewards = ref_rewards.view(-1, self.num_generations).std(dim=1)
        ref_mean_grouped_rewards = ref_mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        ref_std_grouped_rewards = ref_std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        ref_advantages = (ref_rewards - ref_mean_grouped_rewards) / (ref_std_grouped_rewards + 1e-4)

        # Compute refinement loss
        ref_per_token_loss = -(ref_per_token_logps * ref_advantages.unsqueeze(1)) + self.beta * ref_per_token_kl
        # Avoid division by zero for empty completions
        ref_completion_lengths = ref_completion_mask.sum(dim=1).clamp(min=1)
        loss = ((ref_per_token_loss * ref_completion_mask).sum(dim=1) / ref_completion_lengths).mean()
        # Handle NaN/Inf by replacing with zero (preserve requires_grad)
        if torch.isnan(loss) or torch.isinf(loss):
            loss = loss * 0.0

        # Log refinement metrics
        ref_completion_length = self.accelerator.gather_for_metrics(ref_completion_mask.sum(1)).float().mean().item()
        self._metrics["refinement_completion_length"].append(ref_completion_length)
        self._metrics["refinement_reward"].append(self.accelerator.gather_for_metrics(ref_rewards).mean().item())

        ref_mean_kl = ((ref_per_token_kl * ref_completion_mask).sum(dim=1) / ref_completion_lengths).mean()
        self._metrics["refinement_kl"].append(self.accelerator.gather_for_metrics(ref_mean_kl).mean().item())

        return loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Override training_step to support two-step training with proper gradient accumulation.

        For two-step training:
        - Accumulate Pass 1 gradients over gradient_accumulation_steps
        - Step optimizer for Pass 1, update model
        - Run Pass 2 for all cached batches (with updated model)
        - Accumulate Pass 2 gradients
        - Step optimizer for Pass 2 (handled by outer loop)
        """
        model.train()

        # First pass: compute loss, backward, cache completions
        loss_pass1 = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

        if self.args.gradient_accumulation_steps > 1:
            loss_pass1 = loss_pass1 / self.args.gradient_accumulation_steps

        # Backward pass 1
        self.accelerator.backward(loss_pass1)

        if not self.two_step_training:
            return loss_pass1.detach()

        # Two-step training: track accumulation
        self._accumulation_counter += 1
        self._metrics["loss_pass1"].append(loss_pass1.item() * self.args.gradient_accumulation_steps)

        # Check if we've accumulated enough steps
        if self._accumulation_counter < self.args.gradient_accumulation_steps:
            # Not ready for pass 2 yet, return pass 1 loss
            # Return 0 so outer loop doesn't step yet (we handle it ourselves)
            return loss_pass1.detach()

        # Accumulation complete for pass 1 - step optimizer
        if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0:
            self.accelerator.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

        self.optimizer.step()
        # Note: Don't step lr_scheduler here - outer loop will step it once per training step
        self.optimizer.zero_grad()

        # Now run Pass 2 for all cached batches with the updated model
        total_loss_pass2 = 0.0
        for cached_inputs, cached_completions in self._two_step_cache:
            loss_pass2 = self.compute_refinement_loss(model, cached_inputs, cached_completions)

            if self.args.gradient_accumulation_steps > 1:
                loss_pass2 = loss_pass2 / self.args.gradient_accumulation_steps

            self.accelerator.backward(loss_pass2)
            total_loss_pass2 += loss_pass2.item()

        self._metrics["loss_pass2"].append(total_loss_pass2)

        # Clear cache and reset counter
        self._two_step_cache = []
        self._accumulation_counter = 0

        # Return pass 2 loss - outer loop will handle optimizer step for pass 2
        return torch.tensor(total_loss_pass2, device=self.accelerator.device)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Evaluate CER and bCER on the validation set during training."""
        import numpy as np
        from tqdm import tqdm

        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if eval_dataset is None:
            return {}

        model = self.model
        if hasattr(model, 'module'):
            unwrapped_model = model.module
        else:
            unwrapped_model = self.accelerator.unwrap_model(model)

        was_training = unwrapped_model.training
        unwrapped_model.eval()

        is_gc = getattr(unwrapped_model, 'is_gradient_checkpointing', False)
        if is_gc:
            unwrapped_model.gradient_checkpointing_disable()

        from evaluate_csfleurs import compute_cer, compute_bcer, extract_answer
        from utils.rewards import _remove_sp

        metrics = {}

        if self.accelerator.is_main_process:
            model_device = next(unwrapped_model.parameters()).device
            num_eval = len(eval_dataset)
            cer_values = []
            bcer_values = []

            print(f"\n{'='*60}")
            print(f"Validation at step {self.state.global_step} ({num_eval} samples)")
            print(f"{'='*60}")

            for idx in tqdm(range(num_eval), desc="Eval"):
                sample = eval_dataset[idx]

                prompt = sample["prompt"]
                example = {"prompt": prompt}
                prompt_text = maybe_apply_chat_template(example, self.processing_class)["prompt"]

                audio = sample["audio"]
                if isinstance(audio, np.ndarray):
                    audio = audio.astype(np.float32)

                try:
                    inputs = self.processing_class(
                        text=[prompt_text],
                        audio=[audio],
                        sampling_rate=16000,
                        return_tensors="pt",
                        padding=True,
                    )
                    inputs = {k: v.to(model_device) if isinstance(v, torch.Tensor) else v
                              for k, v in inputs.items()}

                    with torch.no_grad():
                        outputs = unwrapped_model.generate(
                            **inputs,
                            max_new_tokens=self.max_completion_length,
                            do_sample=False,
                        )

                    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
                    transcription = self.processing_class.decode(
                        generated_ids, skip_special_tokens=True
                    )
                except Exception as e:
                    logging.warning(f"Eval sample {idx} failed: {e}")
                    continue

                pred = extract_answer(transcription)
                solution = sample.get("solution", "")
                ref = extract_answer(solution)

                language = sample.get("language", "")
                lang = "zh" if language and (
                    "cmn" in language.lower() or "zho" in language.lower()
                ) else "en"

                pred_norm = _remove_sp(pred, lang)
                ref_norm = _remove_sp(ref, lang)

                cer = compute_cer(ref_norm, pred_norm)
                cer_values.append(cer)

                raw_text = sample.get("raw_text", "")
                bcer_result = compute_bcer(raw_text, pred_norm, k=15)
                if bcer_result["bcer"] is not None and bcer_result["boundary_chars"] > 0:
                    bcer_values.append(bcer_result["bcer"])

            avg_cer = sum(cer_values) / len(cer_values) if cer_values else 0.0
            avg_bcer = sum(bcer_values) / len(bcer_values) if bcer_values else None

            print(f"  Eval CER:  {avg_cer:.4f} ({avg_cer * 100:.2f}%)")
            if avg_bcer is not None:
                print(f"  Eval bCER: {avg_bcer:.4f} ({avg_bcer * 100:.2f}%)")
            else:
                print(f"  Eval bCER: N/A (no boundary markers)")
            print(f"{'='*60}\n")

            metrics[f"{metric_key_prefix}_cer"] = avg_cer
            metrics[f"{metric_key_prefix}_bcer"] = avg_bcer if avg_bcer is not None else 0.0

        # Sync across processes
        if self.accelerator.num_processes > 1:
            self.accelerator.wait_for_everyone()

        if is_gc:
            unwrapped_model.gradient_checkpointing_enable()
        if was_training:
            unwrapped_model.train()

        # Log to wandb
        if self.accelerator.is_main_process and metrics:
            if is_wandb_available() and wandb.run is not None:
                wandb.log(metrics, step=self.state.global_step)

        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics
        )

        return metrics

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
