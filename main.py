import copy
import os
import random
from typing import List
from tqdm import tqdm
import fire
import json
import numpy as np
import sys
import math
import torch
import transformers

from datasets import load_dataset
from accelerate import Accelerator
from accelerate.utils import gather_object
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, LlamaForCausalLM, LlamaTokenizer, GenerationConfig
from transformers import TrainerCallback
from transformers import TrainerState
from transformers import Trainer
from data_collator import DataCollatorForSeq2Seq
from new_trainer import newTrainer


def set_seed(seed: int):
    """
    Helper function for setting the random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(
    # model/data params
    base_model: str = "path/to/base_model",
    dataset: str = "game18",
    method: str = "SFT-newseman",
    seed: int = 42,
    is_train: bool = True,
    is_inference: bool = True,
    is_evaluate: bool = True,
    # training hyperparams
    train_batch_size: int = 256,
    micro_batch_size: int = 8,
    num_epochs: int = 5,
    patient: int = 2,
    learning_rate: float = 1e-4,
    cutoff_len: int = 512,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ],
    # inference hyperparams
    inference_batch_size: int = 8,
    num_beams: int = 4,
    num_return_sequences: int = 1,
    sample: int = -1,
    # evaluate hyperparams
    eval_batch_size: int = 16,
    lambda_val: str = "0.85",
    semantic_weight_file: str = "",
):
    set_seed(seed)
    accelerator = Accelerator()
    # data_path
    lora_save_dir = os.path.join("./save_lora_model", dataset, method)

    train_data_path = os.path.join("./data", dataset, 'train', "train_4096.json")
    valid_data_path = os.path.join("./data", dataset, 'train', "valid_512.json")
    test_data_path = os.path.join("./data", dataset, 'test', "test_1000.json")

    # predict_data_path = os.path.join("./data", dataset, "predict", method, "predict_test.json")
    # predict_data_path = os.path.join("./data", dataset, "predict", method, "predict_test_llm.json")  # 消融1
    # predict_data_path = os.path.join("./data", dataset, "predict", method, "predict_test_tfidf.json")  # 消融2
    # predict_data_path = os.path.join("./data", dataset, "predict", method, "predict_test_sasrec.json")  # 消融3
    # predict_data_path = os.path.join("./data", dataset, "predict", method, "predict_test_lambda0.1.json")  # 超参数1
    # predict_data_path = os.path.join("./data", dataset, "predict", method, "predict_test_lambda0.2.json")  # 超参数2
    # predict_data_path = os.path.join("./data", dataset, "predict", method, "predict_test_lambda0.3.json")  # 超参数3
    # predict_data_path = os.path.join("./data", dataset, "predict", method, "predict_test_lambda0.4.json")  # 超参数4
    # predict_data_path = os.path.join("./data", dataset, "predict", method, "predict_test_lambda0.5.json")  # 超参数5
    # predict_data_path = os.path.join("./data", dataset, "predict", method, "predict_test_lambda0.6.json")  # 超参数6
    # predict_data_path = os.path.join("./data", dataset, "predict", method, "predict_test_lambda0.7.json")  # 超参数7
    # predict_data_path = os.path.join("./data", dataset, "predict", method, "predict_test_lambda0.8.json")  # 超参数8
    # predict_data_path = os.path.join("./data", dataset, "predict", method, "predict_test_lambda0.9.json")  # 超参数9

    # result_data_path = os.path.join("./data", dataset, "result", method, "test.json")
    # result_data_path = os.path.join("./data", dataset, "result", method, "test_llm.json")  # 消融1
    # result_data_path = os.path.join("./data", dataset, "result", method, "test_tfidf.json")  # 消融2
    # result_data_path = os.path.join("./data", dataset, "result", method, "test_sasrec.json")  # 消融3
    # result_data_path = os.path.join("./data", dataset, "result", method, "test_lambda0.1.json")  # 超参数1
    # result_data_path = os.path.join("./data", dataset, "result", method, "test_lambda0.2.json")  # 超参数2
    # result_data_path = os.path.join("./data", dataset, "result", method, "test_lambda0.3.json")  # 超参数3
    # result_data_path = os.path.join("./data", dataset, "result", method, "test_lambda0.4.json")  # 超参数4
    # result_data_path = os.path.join("./data", dataset, "result", method, "test_lambda0.5.json")  # 超参数5
    # result_data_path = os.path.join("./data", dataset, "result", method, "test_lambda0.6.json")  # 超参数6
    # result_data_path = os.path.join("./data", dataset, "result", method, "test_lambda0.7.json")  # 超参数7
    # result_data_path = os.path.join("./data", dataset, "result", method, "test_lambda0.8.json")  # 超参数8
    # result_data_path = os.path.join("./data", dataset, "result", method, "test_lambda0.9.json")  # 超参数9

    # redict_train_path = os.path.join("./data", dataset, "predict", method, "predict_train.json")
    result_train_path = os.path.join("./data", dataset, "result", method, "train.json")
    item_embedding_path = os.path.join("./data/", dataset, "item_embedding.pt")
    id2name_path = os.path.join("./data/", dataset, "id2name4Rec.json")
    # seman_path = os.path.join("./data", dataset, 'train', "new_semantic_weight.json")
    # seman_path = os.path.join("./data", dataset, 'train', "new_semantic_weight_llm.json")  # 消融1
    # seman_path = os.path.join("./data", dataset, 'train', "new_semantic_weight_tfidf.json")  # 消融2
    # seman_path = os.path.join("./data", dataset, 'train', "new_semantic_weight_sasrec.json")  # 消融3
    # seman_path = os.path.join("./data", dataset, 'train', "new_semantic_weight_lambda0.1.json")  # 超参数1
    # seman_path = os.path.join("./data", dataset, 'train', "new_semantic_weight_lambda0.2.json")  # 超参数2
    # seman_path = os.path.join("./data", dataset, 'train', "new_semantic_weight_lambda0.3.json")  # 超参数3
    # seman_path = os.path.join("./data", dataset, 'train', "new_semantic_weight_lambda0.4.json")  # 超参数4
    # seman_path = os.path.join("./data", dataset, 'train', "new_semantic_weight_lambda0.5.json")  # 超参数5
    # seman_path = os.path.join("./data", dataset, 'train', "new_semantic_weight_lambda0.6.json")  # 超参数6
    # seman_path = os.path.join("./data", dataset, 'train', "new_semantic_weight_lambda0.7.json")  # 超参数7
    # seman_path = os.path.join("./data", dataset, 'train', "new_semantic_weight_lambda0.8.json")  # 超参数8
    # seman_path = os.path.join("./data", dataset, 'train', "new_semantic_weight_lambda0.9.json")  # 超参数9
    # 替换原本的 predict_data_path, result_data_path, seman_path 的定义
    predict_data_path = os.path.join("./data", dataset, "predict", method, f"predict_test_lambda{lambda_val}.json")
    result_data_path = os.path.join("./data", dataset, "result", method, f"test_lambda{lambda_val}.json")
    if not semantic_weight_file:
        semantic_weight_file = f"new_semantic_weight_lambda{lambda_val}.json"
    seman_path = os.path.join("./data", dataset, 'train', semantic_weight_file)
    if accelerator.is_main_process:
        print(f"lora_save_dir = {lora_save_dir}")
        print(f"train_data_path = {train_data_path}")
        print(f"valid_data_path = {valid_data_path}")
        print(f"test_data_path = {test_data_path}")
        print(f"predict_data_path = {predict_data_path}")
        print(f"result_data_path = {result_data_path}")
        print(f"item_embedding_path = {item_embedding_path}")
        print(f"id2name_path = {id2name_path}")

    # LLM loading
    model, tokenizer = init_model(base_model)

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    # model training
    if is_train:
        callback = MyCallback(tokenizer, valid_data_path, item_embedding_path, id2name_path, 
            num_beams, num_return_sequences, inference_batch_size, eval_batch_size)
        print("### Starting Training ###")
        train(model, tokenizer, train_data_path, method, lora_save_dir, id2name_path, seman_path,
            train_batch_size, micro_batch_size, callback, num_epochs,
            patient, learning_rate, cutoff_len, seed)

    if is_inference:
        print("### Starting Inference on test data ###")
        model, tokenizer = init_model(base_model)
        model = PeftModel.from_pretrained(model, lora_save_dir, torch_dtype=torch.bfloat16)
        model = model.merge_and_unload()
        inference(model, tokenizer, test_data_path, is_save_predict = True, predict_data_path = predict_data_path,
            inference_batch_size=inference_batch_size, num_beams=num_beams, num_return_sequences=num_return_sequences, sample=sample)
        #         这里是核心修改：让推理函数读取训练集，并保存到新的路径
        # print("### Starting Inference on TRAIN data ###") # <--- 修改/添加
        # model, tokenizer = init_model(base_model)
        # # 确保从正确的路径加载你已经微调好的LoRA模型
        # model = PeftModel.from_pretrained(model, lora_save_dir, torch_dtype=torch.bfloat16)
        # model = model.merge_and_unload()
        # inference(model, tokenizer, 
        #           input_data_path = train_data_path, # <--- 修改/添加: 输入改为训练集
        #           is_save_predict = True, 
        #           predict_data_path = predict_train_path, # <--- 修改/添加: 输出到新的predict_train.json
        #           inference_batch_size=inference_batch_size, 
        #           num_beams=num_beams, 
        #           num_return_sequences=num_return_sequences, 
        #           sample=sample)

    if is_evaluate:
        print("### Starting Evaluate on test data ###")
        model, tokenizer = init_model(base_model)
        evaluate(model, tokenizer, item_embedding_path, id2name_path, is_save_eval=True, is_save_predict_truth=True,
            predict_data_path=predict_data_path, result_data_path=result_data_path, eval_batch_size=eval_batch_size)
        # print("### Starting Evaluate on TRAIN data ###") # <--- 修改/添加
        # model, tokenizer = init_model(base_model)
        # evaluate(model, tokenizer, item_embedding_path, id2name_path, 
        #          is_save_eval=True, 
        #          is_save_predict_truth=True,
        #          predict_data_path=predict_train_path, # <--- 修改/添加: 读取新的predict_train.json
        #          result_data_path=result_train_path, # <--- 修改/添加: 评估结果保存到新的train.json
        #          eval_batch_size=eval_batch_size)

def train(
    model,
    tokenizer,
    train_data_path,
    method,
    lora_save_dir,
    id2name_path,
    seman_path,
    train_batch_size,
    micro_batch_size,
    callback,
    num_epochs,
    patient,
    learning_rate,
    cutoff_len,
    seed
):
    
    def tokenize(prompt, add_eos_token=True, split_output=True):
        # 是否仅对output优化
        if split_output:
            prompt_input = prompt.split("### Response:\n")[0] + "### Response:\n"
            result_input = tokenizer(
            prompt_input,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
            )
            prompt_output = prompt.split("### Response:\n")[1]
            result_output = tokenizer(
            prompt_output,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
            )
            result = {"input_ids": result_input["input_ids"] + result_output["input_ids"][1:],
                      "attention_mask": result_input["attention_mask"] + result_output["attention_mask"][1:]}
            if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < cutoff_len
                and add_eos_token
            ):
                result["input_ids"].append(tokenizer.eos_token_id)
                result["attention_mask"].append(1)
            result["label"] = result["input_ids"].copy()
            result["label"][:len(result_input["input_ids"])] = [-100] * (len(result_input["input_ids"]))
        else:
            result = tokenizer(
                prompt,
                truncation=True,
                max_length=cutoff_len,
                padding=False,
                return_tensors=None,
            )
            if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < cutoff_len
                and add_eos_token
            ):
                result["input_ids"].append(tokenizer.eos_token_id)
                result["attention_mask"].append(1)
                result["label"] = result["input_ids"].copy()
        return result


    def generate_and_tokenize_prompt(data_point, output=True):
        full_prompt = generate_prompt(data_point["instruction"], data_point["input"], output=data_point["output"])
        tokenized_full_prompt = tokenize(full_prompt)
        if method == "SFT-pop":
            output_id = int(name2id[data_point["output"][1:-1]])
            freq = freq_dict[output_id]
            # max_freq = max(freq_dict.values())
            # min_freq = min(freq_dict.values())
            # sqrt_freq = math.sqrt(freq)
            # sqrt_max_freq = math.sqrt(max_freq)
            # sqrt_min_freq = math.sqrt(min_freq)
            # # 对数平滑，避免极端权重
            # normalized_freq = (sqrt_freq - sqrt_min_freq) / (sqrt_max_freq - sqrt_min_freq + 1e-8)
            # temperature = 0.4  # 温度参数，调节权重分布
            # weight = 1 / (normalized_freq ** temperature + 0.8)  # 加入平滑项
            alpha = 1
            weight = 1 / (freq**alpha)
            # q20 = np.percentile(list(freq_dict.values()), 20)
            # if freq < q20:
            #     factor = 1.25 + (q20 - freq) / q20 * 0.5
            #     weight *= factor
            # 添加难样本挖掘
            # if freq < np.percentile(list(freq_dict.values()), 30):  # 低频样本
            #     weight *= 1.8  # 增加低频样本权重
            tokenized_full_prompt["weight"] = weight
        if method == "SFT-seman":
            output_id = int(name2id[data_point["output"][1:-1]])
            freq = semantic_dict[output_id]
            alpha = 0.5
            weight = 1 / (freq + 1e-7)**alpha
            tokenized_full_prompt["weight"] = weight
        if method == "SFT-newseman":
            output_id = int(name2id[data_point["output"][1:-1]])
            freq = semantic_dict[output_id]
            #alpha = 0.4
            weight = 1 / float(freq) 
            #weight = 1 / (freq**alpha)
            # q30 = np.percentile(list(semantic_dict.values()), 30)
            # if freq < q30:
            #     factor = 1.5
            #     # factor = 1.35 + (q30 - freq) / q30 * 0.65
            #     weight *= factor
            tokenized_full_prompt["weight"] = weight

        return tokenized_full_prompt

    only_shuffle = True
    gradient_accumulation_steps = train_batch_size // micro_batch_size
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size != 1:
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
    
    
    train_data = load_dataset("json", data_files=train_data_path, split="train")
    if only_shuffle:
        train_data = train_data.shuffle(seed=seed)
    else:
        train_data = (
            train_data.shuffle(seed=seed).select(range(sample)) if sample > -1 else train_data.shuffle(seed=seed)
        )
    

    with open(id2name_path, "r") as file:
        id2name = json.load(file)
    name2id = {}
    for id, name in id2name.items():
        name2id[name] = id
    with open(seman_path, "r") as file:
        semantic_weight = json.load(file)

    all_id = np.array([int(name2id[i[1:-1]]) for i in train_data["output"]])
    unique_ids, counts = np.unique(all_id, return_counts=True)
    unique_ids = [i.item() for i in unique_ids]
    counts = [i.item() for i in counts]
    freq_dict = dict(zip(unique_ids, counts))
    semantic_dict = dict(zip(unique_ids, semantic_weight))




    train_data = train_data.map(lambda x: generate_and_tokenize_prompt(x))
    if method == "SFT-pop" or method == "SFT-seman" or method == "SFT-newseman":
        method = "SFT-weight"
    label_names = None
    if method == "SFT-weight":
        label_names = ["weight"]

    trainer = newTrainer(
        model=model,
        train_dataset=train_data,
        callbacks=[callback],
        args=transformers.TrainingArguments(
            output_dir=lora_save_dir,
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=22,
            lr_scheduler_type="constant_with_warmup",
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            bf16=True,
            tf32=True,
            optim="adamw_torch",
            logging_strategy="steps",
            logging_steps=0.1,
            save_strategy="steps",
            save_steps=(1 / num_epochs),
            # save_total_limit=10,
            save_on_each_node=False,
            log_on_each_node=False,
            # load_best_model_at_end=True,
            ddp_find_unused_parameters=False if (world_size != 1) else None,
            report_to="tensorboard",
            remove_unused_columns=False,
            ddp_backend="nccl",
            local_rank=int(os.environ.get("LOCAL_RANK", -1)),
            seed=seed,
            label_smoothing_factor=0.1,
            label_names = label_names,
        ),
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        train_type = method,
    )
    model.config.use_cache = False  # 在训练时use_cache是useless
    trainer.train()
    model.save_pretrained(lora_save_dir)




class MyCallback(TrainerCallback):
    def __init__(self, tokenizer, valid_data_path, item_embedding_path, id2name_path, 
            num_beams, num_return_sequences, inference_batch_size, eval_batch_size):
        self.tokenizer = tokenizer
        self.valid_data_path = valid_data_path
        self.item_embedding_path = item_embedding_path
        self.id2name_path = id2name_path
        self.num_beams = num_beams
        self.num_return_sequences = num_return_sequences
        self.inference_batch_size = inference_batch_size
        self.eval_batch_size = eval_batch_size
        self.patient = 0
        self.result = []
    def on_train_begin(self, args, state, control, model, **kwargs):
        return control
    def on_epoch_end(self, args, state, control, model, **kwargs):
        predict_valid_data = inference(model, self.tokenizer, self.valid_data_path, is_save_predict=False, 
            num_beams=self.num_beams, num_return_sequences=self.num_return_sequences, inference_batch_size=self.inference_batch_size)
        eval_result = evaluate(model, self.tokenizer, self.item_embedding_path, self.id2name_path, is_save_eval=False, 
            predict_data=predict_valid_data, eval_batch_size=self.eval_batch_size)
        print(f"eval_result = {eval_result}")
        report_result = eval_result["HR@10"]
        if len(self.result) > 0 and report_result <= max(self.result):
            self.patient += 1
            print(f"patient = {self.patient}")
        else:
            self.patient = 0
        self.result.append(report_result)
        if report_result > max(self.result[:-1], default=float('-inf')):  # 如果当前结果是最优的
            self.best_epoch = int(state.epoch)  # 记录最佳的 epoch
            self.best_model_state = model.state_dict()  # 保存模型的参数
        print(f"best epoch: {self.best_epoch}")

        if self.patient >= 2:
            print("early stop!")
            control.should_training_stop = True
        model.train() 
        return control


def inference(
    model,
    tokenizer,
    input_data_path,
    is_save_predict: bool = True,
    predict_data_path: str = "",
    inference_batch_size: int = 8,
    num_beams: int = 4,
    num_return_sequences: int = 1,
    sample: int = -1,
):
    accelerator = Accelerator()
    with open(input_data_path, "r") as f:
        test_data = json.load(f)
    if sample != -1:
        test_data = random.sample(test_data, sample)

    model.eval()

    def generate_output(instructions, inputs=None, max_new_tokens=64, **kwargs):
        prompt = [generate_prompt(instruction, input) for instruction, input in zip(instructions, inputs)]
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(accelerator.device)
        inputs_dict = {key: value for key, value in inputs.items()}

        original_num_elements = inputs_dict["input_ids"].shape[0] * num_beams

        generation_config = GenerationConfig(
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                #do_sample=True,
                num_beams=num_beams,
                num_return_sequences=num_return_sequences,  # 此处应该设置为1
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                output_scores=False,
                **kwargs,
            )
        with torch.no_grad():
            generation_config = generation_config
            generation_output = model.generate(**inputs, generation_config=generation_config)
            output_seq = generation_output.sequences
            output = tokenizer.batch_decode(output_seq, skip_special_tokens=True)
            output = [_.split("Response:\n")[-1] for _ in output]


            real_outputs = output[:original_num_elements]
            real_outputs = [
                real_outputs[i * num_return_sequences : (i + 1) * num_return_sequences] for i in range(len(real_outputs) // num_return_sequences)
            ]
        return real_outputs

    def batch(list, batch_size=inference_batch_size):
        chunk_size = (len(list) - 1) // batch_size + 1
        for i in range(chunk_size):
            yield list[batch_size * i : batch_size * (i + 1)]

    outputs = []
    instructions = [_["instruction"] for _ in test_data]
    inputs = [_["input"] for _ in test_data]
    input_dict = {"instructions": instructions, "inputs": inputs}

    with accelerator.split_between_processes(input_dict) as input_temp:
        outputs = []
        sequences_scores = []

        for batch1 in tqdm(
            zip(batch(input_temp["instructions"]), batch(input_temp["inputs"])),
            total=(len(input_temp["instructions"]) + inference_batch_size - 1) // inference_batch_size,
        ):
            instructions, inputs = batch1
            output = generate_output(instructions, inputs)
            outputs.extend(output)

    outputs = gather_object(outputs)
    for i, _ in tqdm(enumerate(test_data)):
        test_data[i]["predict"] = outputs[i]
    
    if not is_save_predict:
        return test_data
    # 调试打印：在写文件前插入
    # print("DEBUG: is_save_predict=", is_save_predict)
    # print("DEBUG: accelerator.is_main_process=", accelerator.is_main_process)
    # print("DEBUG: predict_data_path=", repr(predict_data_path))
    # print("DEBUG: len(test_data)=", len(test_data))
    # print("DEBUG: local outputs length=", len(outputs))
    # 如果 accelerate 提供 wait 方法
    try:
        accelerator.wait_for_everyone()
        print("DEBUG: wait_for_everyone completed")
    except Exception:
        pass
    if accelerator.is_main_process:
        if not predict_data_path:
            raise ValueError('predict_data_path is empty')
        folder_path = os.path.dirname(predict_data_path)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        with open(predict_data_path, "w") as f:
            json.dump(test_data, f, indent=4)

def evaluate(
    model,
    tokenizer,
    item_embedding_path,
    id2name_path,
    is_save_eval: bool = True,
    is_save_predict_truth: bool = True,
    predict_data_path: str = "",
    predict_data: list = [],
    result_data_path: str = "",
    eval_batch_size: int = 16,
):
    accelerator = Accelerator()
    def batch(list, batch_size):
        chunk_size = (len(list) - 1) // batch_size + 1
        for i in range(chunk_size):
            yield list[batch_size * i : batch_size * (i + 1)]
    def compute_metrics(rank_tensor, test_data, item_dict, topk_list=[1, 5, 10, 20, 50]):
        NDCG, HR = [], []

        target_items = [item["output"][1:-1] for item in test_data]
        target_item_ids = torch.tensor([item_dict[item] for item in target_items], device="cuda")
        target_item_ranks = rank_tensor[torch.arange(rank_tensor.size(0)), target_item_ids]
        rank_list_tensor = target_item_ranks
        reciprocal_ranks = 1.0 / (rank_list_tensor+1).float()
        #mrr = torch.mean(reciprocal_ranks)
        for k in topk_list:
            Hit_num = (rank_list_tensor < k).sum().item()
            HR.append(Hit_num / len(test_data))

            mask = rank_list_tensor < k
            NDCG_num = 1 / torch.log(rank_list_tensor[mask] + 2)
            NDCG.append(NDCG_num.sum().item() / len(test_data) / (1.0 / math.log(2)))

        result_dict = dict()
        for i in range(len(topk_list)):
            result_dict["NDCG@" + str(topk_list[i])] = NDCG[i]

        for i in range(len(topk_list)):
            result_dict["HR@" + str(topk_list[i])] = HR[i]

        return result_dict, rank_list_tensor


    def generate_result_file(
        model, tokenizer, predict_path, test_data,  batch_size, item_embedding_table, item_dict, id2name
    ):
        if not test_data:
            f = open(predict_path, "r")
            test_data = json.load(f)
            f.close()
        for item in test_data:
            item["predict_truth_item"] = []
        beam_num = len(test_data[0]["predict"])
        for num in range(beam_num):
            text = [_["predict"][num].strip('"').strip(" ") for _ in test_data]
            with torch.no_grad():
                predict_embeddings = []
                for batch_input in tqdm(batch(text, batch_size=batch_size), total=len(text) // batch_size + 1):
                    inputs = tokenizer(batch_input, return_tensors="pt", padding=True).to("cuda")
                    outputs = model(inputs.input_ids, attention_mask=inputs.attention_mask, output_hidden_states=True)
                    hidden_states = outputs.hidden_states
                    predict_embeddings.append(hidden_states[-1][:, -1, :].detach())
                predict_embeddings = torch.cat(predict_embeddings, dim=0).to(dtype=torch.bfloat16)  # 5000 x 32000
            dist = torch.cdist(predict_embeddings.cuda(), item_embedding_table.cuda(), p=2)  # 5000 x item_num
            rank = dist.argsort(dim=-1).argsort(dim=-1)  # 两次argsort后，rank的每一行的每个位置对应每个item的排名

            zero_row_indices = (item_embedding_table==0).all(axis=1).nonzero().squeeze().tolist()
            if isinstance(zero_row_indices, int):
                zero_row_indices = [zero_row_indices] # 把它包装成一个列表
            predict_id = []
            sort_values, sort_indices = rank.sort(dim=1)
            for indice in sort_indices:
                for i in range(len(zero_row_indices)+1):
                    if indice[i].item() not in zero_row_indices:
                        break
                predict_id.append(indice[i].item())
            for i,id in enumerate(predict_id):
                truth_id = item_dict.get(test_data[i]["output"][1:-1], -1)
                test_data[i]["predict_truth_item"].append(id2name[id])

        result_dict, _ = compute_metrics(rank, test_data, item_dict)
        if is_save_eval:
            if not result_data_path:
                raise ValueError('result_data_path is empty')
            folder_path = os.path.dirname(result_data_path)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            print(result_dict)
            f = open(result_data_path, "w")
            json.dump(result_dict, f, indent=4)
            if is_save_predict_truth:
                predict_data_path_truth = predict_data_path.split(".json")[0] + "_truth.json"
                f = open(predict_data_path_truth, "w")
                json.dump(test_data, f, indent=4)

        else:
            return result_dict


    item_embedding_table = torch.load(item_embedding_path)
    with open(id2name_path, "r") as file:
        data = json.load(file)
    name2id_dict = {v: int(k) for k, v in data.items()}
    id2name_dict = {int(k): v for k, v in data.items()}
    dist = generate_result_file(
        model,
        tokenizer,
        predict_data_path,
        predict_data,
        eval_batch_size,
        item_embedding_table,
        name2id_dict,
        id2name_dict,
        )
    
    if dist is not None:
        return dist

def init_model(base_model):
    bnb_config = None 
    model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map={"": int(os.environ.get("LOCAL_RANK") or 0)},
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.add_special_tokens({"pad_token": "<pad>"})
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    tokenizer.padding_side = "left"
    return model, tokenizer

def generate_prompt(instruction, input, output=""):
    if not output:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""
    else:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

if __name__ == "__main__":
    fire.Fire(main)
