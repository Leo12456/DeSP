from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from transformers import PreTrainedTokenizerBase


@dataclass
class DataCollatorForSeq2Seq:
    """Pad causal-LM inputs while preserving DeSP's custom labels and weights."""

    tokenizer: PreTrainedTokenizerBase
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    padding: bool = True
    label_pad_token_id: int = -100

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        features = [dict(feature) for feature in features]
        label_key = "label" if "label" in features[0] else "labels"
        labels = [feature.pop(label_key) for feature in features] if label_key in features[0] else None
        weights = [feature.pop("weight") for feature in features] if "weight" in features[0] else None

        model_features = [
            {key: feature[key] for key in ("input_ids", "attention_mask") if key in feature}
            for feature in features
        ]
        batch = self.tokenizer.pad(
            model_features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        if labels is not None:
            sequence_length = batch["input_ids"].shape[1]
            padding_side = self.tokenizer.padding_side
            padded_labels = []
            for label in labels:
                remainder = sequence_length - len(label)
                if remainder < 0:
                    label = label[:sequence_length]
                    remainder = 0
                pad = [self.label_pad_token_id] * remainder
                if padding_side == "right":
                    padded_labels.append(label + pad)
                else:
                    padded_labels.append(pad + label)
            batch["label"] = torch.tensor(padded_labels, dtype=torch.long)

        if weights is not None:
            batch["weight"] = torch.tensor(weights, dtype=torch.float32)

        return batch
