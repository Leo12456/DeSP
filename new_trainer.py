import math
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import torch
import torch.nn as nn
import numpy as np
import os
import json
from filelock import FileLock
from scipy.special import lambertw
from datasets import Dataset
from dataclasses import dataclass
import torch.nn.functional as F
from transformers import DataCollator, PreTrainedModel, PreTrainedTokenizerBase, Trainer, TrainingArguments
from transformers.trainer_callback import TrainerCallback
from transformers import TrainerState




class newTrainer(Trainer):
    def __init__(
        self,
        ref_model: Union[PreTrainedModel, nn.Module] = None,
        alpha: float = 1,
        beta: float = 1,
        train_type: str = "SFT",
        args: TrainingArguments = None,
        **kwargs, 
    ):
        # ------------------- 修复开始 -------------------
        # 在调用 super().__init__ 之前，从 kwargs 中弹出自定义参数
        # 否则它们会被传递给 Trainer 基类，导致 TypeError
        self.rank_lambda = kwargs.pop("rank_lambda", 0.0) # 0.0 是默认值
        self.rank_gamma = kwargs.pop("rank_gamma", 0.0)   # 0.0 是默认值
        # ------------------- 修复结束 ------------------- 
        self.ref_model = ref_model
        self.alpha = alpha,
        self.beta = beta,
        self.train_type = train_type,
        if not isinstance(self.alpha, float):
            self.alpha = self.alpha[0]   
        if not isinstance(self.beta, float):
            self.beta = self.beta[0]   
        if not isinstance(self.train_type, str):
            self.train_type = self.train_type[0]
        super().__init__(
            args=args,
            **kwargs, 
        )

    def compute_loss(self, model, inputs, num_items_in_batch=None):
        '''
        最近修改地方出现TypeError情况没传入num_items_in_batch
        '''
        if num_items_in_batch is None:
            num_items_in_batch = len(inputs.get('input_ids', []))

        truth_inputs = {"input_ids":inputs["input_ids"],
                    "attention_mask":inputs["attention_mask"],
                    "labels":inputs["label"]}
        
        
        if self.train_type == "SFT":
            SFT_loss = self.compute_loss_SFT(model, truth_inputs)
            return SFT_loss
        elif self.train_type == "SFT-weight":
            SFT_loss = self.compute_loss_SFT(model, truth_inputs, weight=inputs["weight"])
            return SFT_loss
        else:
            SFT_loss = self.compute_loss_SFT(model, truth_inputs)
            ref_inputs = {"input_ids":inputs["ref_1_input_ids"],
                    "attention_mask":inputs["ref_1_attention_mask"],
                    "labels":inputs["ref_1_labels"]}
            ref_loss = self.compute_loss_SFT(model, ref_inputs)
            if self.train_type == "SSFT":
                loss = 0.5*SFT_loss + 0.5*ref_loss
                print(loss)
            
            elif self.train_type == "DPO":
                truth_inputs = {"input_ids":inputs["input_ids"],
                            "attention_mask":inputs["attention_mask"],
                            "labels":inputs["label"]}
                ref_inputs = {"input_ids":inputs["ref_1_input_ids"],
                            "attention_mask":inputs["ref_1_attention_mask"],
                            "labels":inputs["ref_1_labels"]}
                with torch.no_grad():
                    SFT_loss_ref_model = self.compute_loss_SFT(self.ref_model, truth_inputs)
                    ref_loss_ref_model = self.compute_loss_SFT(self.ref_model, ref_inputs)
                #print(SFT_loss, ref_loss, SFT_loss_ref_model, ref_loss_ref_model)
                DPO_logits = -SFT_loss + ref_loss + SFT_loss_ref_model - ref_loss_ref_model
                loss = -F.logsigmoid(self.beta * DPO_logits)
                #exit()
            if self.train_type == "SDFT":
                curent_epoch = math.floor(self.state.epoch) + 1
                my_callback = self.callback_handler.callbacks[2]
                dist = my_callback.dist[curent_epoch-1]
                if my_callback.dist_origin != -1:
                    dist_origin = my_callback.dist_origin
                else:
                    dist_origin = my_callback.dist[0]
                
                if curent_epoch != 1:
                    dist = 0.6*dist_origin
                all_epoch = self.args.num_train_epochs
                # 线性衰减
                epoch_lambda = 1 - self.rank_lambda * ((dist_origin-dist)/dist_origin) ** 2
                if epoch_lambda < 0:
                    epoch_lambda = 0
                # 指数衰减
                #epoch_lambda = math.e ** -(self.rank_lambda * (curent_epoch-1))
                # super loss
                #epoch_lambda = math.e ** (self.rank_lambda * (dist/dist_origin-1)) 
                print(epoch_lambda)
                loss = (1-epoch_lambda) * SFT_loss + epoch_lambda * ref_loss  

        return loss

    def compute_loss_SFT(self, model, inputs, weight=None):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)

        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            #unwrapped_model = self.accelerator.unwrap_model(model)
            #model_name = unwrapped_model.base_model.model._get_name()
            label_smoother = LabelSmoother(epsilon=self.args.label_smoothing_factor)
            loss = label_smoother(outputs, labels, shift_labels=True, weight=weight)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return loss


    def get_batch_logps(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
    ) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
            average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")
        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_mask = labels != self.label_pad_token_id

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == self.label_pad_token_id] = 0

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)

    


@dataclass
class LabelSmoother:
    """
    Adds label-smoothing on a pre-computed output from a Transformers model.

    Args:
        epsilon (`float`, *optional*, defaults to 0.1):
            The label smoothing factor.
        ignore_index (`int`, *optional*, defaults to -100):
            The index in the labels to ignore when computing the loss.
    """

    epsilon: float = 0.1
    ignore_index: int = -100

    def __call__(self, model_output, labels, shift_labels=False, weight=None):
        logits = model_output["logits"] if isinstance(model_output, dict) else model_output[0]
        if shift_labels:
            logits = logits[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()

        log_probs = -nn.functional.log_softmax(logits, dim=-1)
        if labels.dim() == log_probs.dim() - 1:
            labels = labels.unsqueeze(-1)

        padding_mask = labels.eq(self.ignore_index)
        # In case the ignore_index is -100, the gather will fail, so we replace labels by 0. The padding_mask
        # will ignore them in any case.
        labels = torch.clamp(labels, min=0)
        nll_loss = log_probs.gather(dim=-1, index=labels)
        # works for fp16 input tensor too, by internally upcasting it to fp32
        smoothed_loss = log_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)
        
        nll_loss.masked_fill_(padding_mask, 0.0)
        smoothed_loss.masked_fill_(padding_mask, 0.0)

        # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
        num_active_elements = padding_mask.numel() - padding_mask.long().sum()

        if weight is None:
            nll_loss = nll_loss.sum() / num_active_elements
        else:
            # # --- SNIPS 核心逻辑开始 ---
            # 1. 获取batch_size
            num_samples_in_batch = weight.size(0)
            # 2. 计算batch内权重的总和
            sum_of_weights = torch.sum(weight)
            # 3. 防止除以零
            if sum_of_weights > 1e-8:
            # 4. 对权重进行标准化 (核心步骤)
                 normalized_weight = weight * (num_samples_in_batch / sum_of_weights)
            else:
            # 如果权重和过小，则退化为不加权
                normalized_weight = torch.ones_like(weight)
        # # --- SNIPS 核心逻辑结束 ---
        # #    这对应于 SNIPS 公式中的分母: Σ(1/P_ui)
        #     sum_of_weights = torch.sum(weight)
            
        #     # 2. 为数值稳定性进行检查
        #     if sum_of_weights > 1e-8:
        #         # 3. 计算分子: 每个token的损失乘以其对应样本的权重，然后求和
        #         #    nll_loss * weight.view(-1, 1, 1) -> 这是加权后的 per-token loss
        #         #    .sum() -> 求和得到SNIPS公式的分子: Σ(δ * 1/P_ui)
        #         weighted_nll_loss_numerator = (nll_loss * weight.view(-1, 1, 1)).sum()
                
        #         # 4. 应用SNIPS公式(分子/分母)，直接得到最终的加权平均损失
        #         #    这一步直接计算出正确的 nll_loss，替换了您原有的实现
        #         nll_loss = weighted_nll_loss_numerator / sum_of_weights
        #     else:
        #         # 如果权重和过小 (例如，一个批次都是权重为0的样本)，
        #         # 则退化为不加权的 token 平均损失，以防止除以零。
        #         nll_loss = nll_loss.sum() / num_active_elements
            weight_nll_loss = nll_loss * normalized_weight.view(-1, 1, 1)
            nll_loss = weight_nll_loss.sum() / (num_active_elements)
        smoothed_loss = smoothed_loss.sum() / (num_active_elements * log_probs.shape[-1])
        return (1 - self.epsilon) * nll_loss + self.epsilon * smoothed_loss