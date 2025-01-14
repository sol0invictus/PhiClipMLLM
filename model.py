import torch
import torch.nn as nn
import os
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoProcessor,
    CLIPVisionModel,
)
from peft import get_peft_model, LoraConfig  # type: ignore
from typing import List, Union
from typing import Optional
from pathlib import Path

_PLACEHOLDER_IMAGE = "<|placeholder1|>"
_PLACEHOLDER_INSTRUCT = "<|placeholder2|>"

import torch
import torch.nn as nn


class MultimodalProjector(nn.Module):
    def __init__(
        self,
        image_embedding_dim,
        text_embedding_dim,
        hidden_dim=None,
        activation=nn.ReLU,
        normalization=True,
    ):
        """
        A projection layer to align image embeddings with the text embedding space.

        Args:
            image_embedding_dim (int): Dimensionality of image embeddings from the image model.
            text_embedding_dim (int): Dimensionality of text embeddings in the text model.
            hidden_dim (int, optional): Optional intermediate hidden dimension for the projector. Defaults to None.
            activation (callable, optional): Activation function to apply after the projection. Defaults to nn.ReLU.
            normalization (bool, optional): Whether to normalize the output to unit length. Defaults to True.
        """
        super(MultimodalProjector, self).__init__()

        layers = []

        if hidden_dim is not None:
            # If a hidden dimension is specified, use a two-layer projection
            layers.append(nn.Linear(image_embedding_dim, hidden_dim))
            if activation is not None:
                layers.append(activation())
            layers.append(nn.Linear(hidden_dim, text_embedding_dim))
        else:
            # If no hidden dimension, use a single linear layer
            layers.append(nn.Linear(image_embedding_dim, text_embedding_dim))

        self.projector = nn.Sequential(*layers)
        self.normalization = normalization

        # Apply initialization
        self._initialize_weights()

    def forward(self, image_embeddings):
        """
        Forward pass for the multimodal projector.

        Args:
            image_embeddings (torch.Tensor): Image embeddings of shape [batch_size, image_embedding_dim].

        Returns:
            torch.Tensor: Projected embeddings of shape [batch_size, text_embedding_dim].
        """
        projected = self.projector(image_embeddings)  # [batch_size, text_embedding_dim]

        if self.normalization:
            # Normalize to unit length
            projected = nn.functional.normalize(projected, p=2, dim=-1)

        return projected

    def _initialize_weights(self):
        """
        Initializes weights of linear layers with He or Xavier initialization.
        """
        for module in self.projector:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(
                    module.weight, nonlinearity="relu"
                )  # He initialization
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


class PhiClipMLLM(nn.Module):

    def __init__(
        self,
        language_model="",
        vision_model="",
        adapter_dim=512,
        lora_config: List[LoraConfig] = None,
        dtype=torch.float16,
    ):
        super(PhiClipMLLM, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vision_model = CLIPVisionModel.from_pretrained(
            vision_model, torch_dtype=dtype
        )
        self.vision_model.to(self.device)
        self.vision_processor = AutoProcessor.from_pretrained(vision_model)
        self.text_model = AutoModelForCausalLM.from_pretrained(
            language_model, torch_dtype=dtype, low_cpu_mem_usage=True
        )
        self.text_model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(language_model)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        self.instuction_training = True
        self.dtype = dtype
        self.vision_adapter = (
            MultimodalProjector(
                image_embedding_dim=1024,
                text_embedding_dim=3072,
                hidden_dim=adapter_dim,
                activation=nn.GELU,
                normalization=True,
            )
            .to(dtype)
            .to(self.device)
        )
        self.prime_model(stage="one")

    def setup_lora(self, lora_config: List[LoraConfig]):
        self.text_model = get_peft_model(self.text_model, lora_config[0])
        self.vision_model = get_peft_model(self.vision_model, lora_config[1])
        # Freeze everything except the adapter
        self.prime_model(stage="two")

    def prime_model(self, stage="one"):
        if stage == "one":
            # Freeze all models except the vision adapter
            for param in self.vision_model.parameters():
                param.requires_grad = False
            for param in self.text_model.parameters():
                param.requires_grad = False

            # Only train the vision adapter
            for param in self.vision_adapter.parameters():
                param.requires_grad = True

        else:
            # Unfreeze all models
            for param in self.vision_model.parameters():
                param.requires_grad = False
            for param in self.text_model.parameters():
                param.requires_grad = False
            for param in self.vision_adapter.parameters():
                param.requires_grad = True

            # Unfreeze LoRA layers
            for name, param in self.vision_model.named_parameters():
                if "lora" in name:
                    param.requires_grad = True
            for name, param in self.text_model.named_parameters():
                if "lora" in name:
                    param.requires_grad = True

    def create_position_ids(self, attention_mask):
        # Create position ids that ignore padding tokens
        position_ids = attention_mask.cumsum(dim=1) - 1
        position_ids = position_ids.masked_fill(attention_mask == 0, 0)
        return position_ids

    def forward(self, batch):

        input_texts = batch["input_texts"]
        input_images = batch["input_images"]
        labels = batch.get("labels", None)

        # Generate text and image embeddings
        text_embeddings, img_token_mask, attention_mask = self.encode_text(input_texts)
        image_embeddings = self.encode_image(input_images)

        text_embeddings = text_embeddings.to(self.dtype)
        image_embeddings = image_embeddings.to(self.dtype)

        adapted_image_embeddings = self.vision_adapter(image_embeddings)
        # Combine embeddings by replacing the [placeholder] token with image embeddings
        combined_embedding = self.combine_text_image(
            text_embeddings, adapted_image_embeddings, img_token_mask
        )
        # Add position embedding back
        expanded_attention_masks = []

        for batch_idx, mask in enumerate(img_token_mask):
            img_position = mask.nonzero(as_tuple=True)[0][0]
            prefix_mask = attention_mask[batch_idx, :img_position]
            suffix_mask = attention_mask[batch_idx, (img_position + 1) :]
            image_mask = torch.ones(
                adapted_image_embeddings.size(1), device=attention_mask.device
            )
            expanded_mask = torch.cat([prefix_mask, image_mask, suffix_mask], dim=0)
            expanded_attention_masks.append(expanded_mask)
        expanded_attention_mask = torch.stack(expanded_attention_masks)

        position_ids = self.create_position_ids(expanded_attention_mask)

        # Compute logits
        outputs = self.text_model(
            inputs_embeds=combined_embedding,
            attention_mask=expanded_attention_mask,
            position_ids=position_ids,
        )

        # Compute logits
        logits = outputs.logits

        result = {"logits": logits}

        if labels is not None:
            labels = [
                text.replace("<|placeholder1|>", "<|placeholder1|>" * 257)
                for text in labels
            ]
            # Shift labels to align with predicted outputs for loss computation
            label_tokens = self.encode_text(
                labels, token_only=True, instructions=self.instuction_training
            )
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = label_tokens[..., 1:].contiguous()
            # Define loss function (cross-entropy for language modeling)
            loss = self.loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            result["loss"] = loss

        return result  # Output multimodal embeddings for downstream tasks

    def __call__(self, input_texts, input_images, labels=None):
        return self.forward(
            {"input_texts": input_texts, "input_images": input_images, "labels": labels}
        )

    def encode_text(self, input_text, token_only=False, instructions=False):

        tokenized = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            max_length=512,
            return_attention_mask=True,
            truncation=True,
        )
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        if token_only:
            if instructions:
                instruction_placeholder_positions = (
                    input_ids
                    == self.tokenizer.convert_tokens_to_ids(_PLACEHOLDER_INSTRUCT)
                ).nonzero(as_tuple=False)
                batch_size, seq_len = input_ids.size()
                mask = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
                placeholder_indices = instruction_placeholder_positions[:, 1]
                mask = mask <= placeholder_indices.unsqueeze(1)
                modified_sequences = input_ids.clone()
                modified_sequences[mask] = self.tokenizer.pad_token_id
                return modified_sequences.to(self.device)
            else:
                return input_ids.to(self.device)
        img_token_id = self.tokenizer.convert_tokens_to_ids(_PLACEHOLDER_IMAGE)
        img_token_mask = input_ids == img_token_id  # [batch_size, seq_len]
        text_embeddings = self.text_model.get_input_embeddings()(
            input_ids.to(self.device)
        )
        return text_embeddings, img_token_mask, attention_mask.to(self.device)

    def encode_image(self, input_images):
        inputs = self.vision_processor(images=input_images, return_tensors="pt")
        outputs = self.vision_model(**inputs.to(self.device)).last_hidden_state
        return outputs

    def combine_text_image(self, text_embeddings, image_embeddings, img_token_mask):
        """
        Combines text and image embeddings by inserting the full image token sequence at the <img> token position.
        One image per sequence only.
        """
        import torch

        batch_size, seq_len, hidden_dim = text_embeddings.shape
        _, img_seq_len, _ = image_embeddings.shape

        output_embeddings = []

        for batch_idx, mask in enumerate(img_token_mask):
            # Get position of single <img> token
            img_position = mask.nonzero(as_tuple=True)[0][0]

            # Split text at image position
            prefix = text_embeddings[batch_idx, :img_position]
            suffix = text_embeddings[batch_idx, (img_position + 1) :]

            # Combine: prefix + image + suffix
            batch_output = torch.cat(
                [prefix, image_embeddings[batch_idx], suffix], dim=0
            )

            output_embeddings.append(batch_output)

        return torch.stack(output_embeddings)

    def save_checkpoint(self, save_path: Union[str, Path]):
        """
        Save the model's complete state including configurations.
        """
        save_path = Path(save_path)
        os.makedirs(save_path.parent, exist_ok=True)

        # Get the original processor name from the processor's config name
        vision_processor_name = self.vision_processor.tokenizer.name_or_path

        uses_lora = hasattr(self.text_model, "peft_config")
        print("LORA: ", uses_lora)
        if uses_lora:
            # Merge and get fused states
            vision_state, text_state, adapter_state = self.merge_and_save_lora()
        else:
            # Get regular states
            vision_state = self.vision_model.state_dict()
            text_state = self.text_model.state_dict()
            adapter_state = self.vision_adapter.state_dict()

        vision_processor_name = self.vision_processor.tokenizer.name_or_path

        checkpoint = {
            "vision_model_state": vision_state,
            "vision_adapter_state": adapter_state,
            "text_model_state": text_state,
            "vision_config": self.vision_model.config,
            "text_model_config": self.text_model.config,
            "vision_processor_name": vision_processor_name,
            "vision_adapter_config": {
                "image_embedding_dim": self.vision_adapter.projector[0].in_features,
                "text_embedding_dim": self.vision_adapter.projector[-1].out_features,
                "hidden_dim": (
                    self.vision_adapter.projector[0].out_features
                    if len(self.vision_adapter.projector) > 1
                    else None
                ),
                "activation": (
                    type(self.vision_adapter.projector[1])
                    if len(self.vision_adapter.projector) > 2
                    else None
                ),
                "normalization": self.vision_adapter.normalization,
            },
        }
        torch.save(checkpoint, save_path)
        self.tokenizer.save_pretrained(save_path.parent)
        print(f"Checkpoint saved at {save_path}")

    def load_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        map_location: Optional[Union[str, torch.device]] = None,
        strict: bool = True,
    ) -> None:
        """
        Load weights from a checkpoint into an existing model instance.

        Args:
            checkpoint_path (Union[str, Path]): Path to the checkpoint file or directory
            map_location (Optional[Union[str, torch.device]]): Device to map the checkpoint to
            strict (bool): Whether to strictly enforce that the keys in state_dict match

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            RuntimeError: If checkpoint loading fails
        """
        try:
            checkpoint_path = Path(checkpoint_path)
            if checkpoint_path.is_dir():
                checkpoint_file = checkpoint_path / "model.pt"
            else:
                checkpoint_file = checkpoint_path

            if not checkpoint_file.exists():
                raise FileNotFoundError(f"Checkpoint not found at {checkpoint_file}")

            # Use current device if map_location not specified
            if map_location is None:
                map_location = self.device

            # Load checkpoint
            checkpoint = torch.load(checkpoint_file, map_location=map_location)

            # Load model states
            self.vision_model.load_state_dict(
                checkpoint["vision_model_state"], strict=strict
            )
            self.vision_adapter.load_state_dict(
                checkpoint["vision_adapter_state"], strict=strict
            )
            self.text_model.load_state_dict(
                checkpoint["text_model_state"], strict=strict
            )

            # Load tokenizer if it exists in checkpoint directory
            tokenizer_path = checkpoint_path.parent
            if (tokenizer_path / "tokenizer_config.json").exists():
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

            # Verify processor compatibility
            current_processor_type = self.vision_processor.tokenizer.name_or_path
            checkpoint_processor_type = checkpoint["vision_processor_name"]
            if current_processor_type != checkpoint_processor_type:
                print(
                    f"Warning: Current processor type ({current_processor_type}) "
                    f"differs from checkpoint ({checkpoint_processor_type})"
                )

            print(f"Successfully loaded checkpoint from {checkpoint_file}")

        except Exception as e:
            raise RuntimeError(f"Error loading checkpoint: {str(e)}") from e

    def merge_and_save_lora(self):
        """
        Merges LoRA weights with base model weights and returns merged state dicts.
        """
        # Merge vision model weights
        self.vision_model.merge_and_unload()
        merged_vision_state = self.vision_model.state_dict()

        # Merge text model weights
        self.text_model.merge_and_unload()
        merged_text_state = self.text_model.state_dict()

        # Merge vision adapter weights
        # self.vision_adapter.merge_and_unload()
        merged_adapter_state = self.vision_adapter.state_dict()

        return merged_vision_state, merged_text_state, merged_adapter_state

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: Union[str, Path],
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float16,
    ) -> "PhiClipMLLM":
        """
        Load a PhiClipMLLM instance directly from a checkpoint without loading base models first.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.is_dir():
            checkpoint_file = checkpoint_path / "model.pt"
        else:
            checkpoint_file = checkpoint_path

        # Load checkpoint
        checkpoint = torch.load(checkpoint_file, map_location="cuda")
        print("Checkpoint loaded")
        # Create empty model instance
        model = cls.__new__(cls)
        nn.Module.__init__(model)
        model.dtype = dtype
        # Initialize basic attributes
        model.device = device
        model.instuction_training = True

        # Clean state dict keys
        checkpoint["vision_model_state"] = {
            k.replace("base_model.model.", ""): v
            for k, v in checkpoint["vision_model_state"].items()
        }
        checkpoint["text_model_state"] = {
            k.replace("base_model.model.", ""): v
            for k, v in checkpoint["text_model_state"].items()
        }
        checkpoint["vision_adapter_state"] = {
            k.replace("base_model.model.", ""): v
            for k, v in checkpoint["vision_adapter_state"].items()
        }

        # Load vision model
        model.vision_model = CLIPVisionModel(config=checkpoint["vision_config"])
        model.vision_model.load_state_dict(checkpoint["vision_model_state"])
        model.vision_model.to(device=device, dtype=dtype)
        print("Vision model loaded")
        # Load vision processor
        model.vision_processor = AutoProcessor.from_pretrained(
            checkpoint["vision_processor_name"]
        )

        # Load text model
        model.text_model = AutoModelForCausalLM.from_config(
            checkpoint["text_model_config"]
        )
        model.text_model.load_state_dict(checkpoint["text_model_state"])
        model.text_model.to(device=device, dtype=dtype)
        print("Text model loaded")
        # Load tokenizer
        model.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path.parent)

        # Initialize loss function
        model.loss_fct = nn.CrossEntropyLoss(ignore_index=model.tokenizer.pad_token_id)

        # Load vision adapter
        adapter_config = checkpoint["vision_adapter_config"]
        model.vision_adapter = MultimodalProjector(**adapter_config)
        model.vision_adapter.load_state_dict(checkpoint["vision_adapter_state"])
        model.vision_adapter.to(device=device, dtype=dtype)
        print("Vision adapter loaded")
        del checkpoint
        return model


def generate(
    model: PhiClipMLLM,
    input_text: str,
    images: List[str] = None,
    max_length: int = 300,
    temperature: float = 0.0,
    top_p: float = 0.9,
    top_k: int = 50,
    num_return_sequences: int = 1,
    do_sample: bool = False,
) -> List[str]:
    """
    Generate text using the MLLM model given input text and optional images.

    Args:
        model: The PhiClipMLLM model instance
        input_text: Input text prompt containing placeholders for images if needed
        images: Optional list of image paths
        max_length: Maximum length of generated text
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        num_return_sequences: Number of sequences to generate
        do_sample: Whether to use sampling or greedy decoding

    Returns:
        List of generated text sequences
    """
    model.eval()
    with torch.no_grad():
        # Process images if provided
        input_images = None
        if images is not None:
            from PIL import Image

            input_images = [Image.open(img_path) for img_path in images]

        # Generate text and image embeddings
        text_embeddings, img_token_mask, attention_mask = model.encode_text(input_text)
        image_embeddings = model.encode_image(input_images)

        adapted_image_embeddings = model.vision_adapter(image_embeddings)
        # Combine embeddings by replacing the [placeholder] token with image embeddings
        combined_embedding = model.combine_text_image(
            text_embeddings, adapted_image_embeddings, img_token_mask
        )
        # Add position embedding back
        expanded_attention_masks = []

        for batch_idx, mask in enumerate(img_token_mask):
            img_position = mask.nonzero(as_tuple=True)[0][0]
            prefix_mask = attention_mask[batch_idx, :img_position]
            suffix_mask = attention_mask[batch_idx, (img_position + 1) :]
            image_mask = torch.ones(
                adapted_image_embeddings.size(1), device=attention_mask.device
            )
            expanded_mask = torch.cat([prefix_mask, image_mask, suffix_mask], dim=0)
            expanded_attention_masks.append(expanded_mask)
        expanded_attention_mask = torch.stack(expanded_attention_masks)

        position_ids = model.create_position_ids(expanded_attention_mask)
        # Compute logits

        # Set up generation parameters
        gen_config = {
            "max_length": max_length,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "num_return_sequences": num_return_sequences,
            "do_sample": do_sample,
            "pad_token_id": model.tokenizer.pad_token_id,
            "eos_token_id": model.tokenizer.eos_token_id,
        }

        # Generate
        outputs = model.text_model.generate(
            inputs_embeds=combined_embedding,
            attention_mask=expanded_attention_mask,
            position_ids=position_ids,
            repetition_penalty=1.2,
            **gen_config,
        )

        # Decode outputs
        generated_texts = model.tokenizer.batch_decode(
            outputs, skip_special_tokens=True
        )

        return generated_texts
