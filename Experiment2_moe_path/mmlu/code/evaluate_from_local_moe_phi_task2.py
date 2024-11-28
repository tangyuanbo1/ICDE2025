import csv
import json
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"]='2,3'
import torch
import random
import transformers
import time
import re
from vllm import LLM, SamplingParams
from tqdm import tqdm
import logging
import sys
from datasets import load_dataset
import pandas as pd


choices = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"]
max_model_length = 4096
max_new_tokens = 2048

from types import MethodType
from typing import Optional
import torch.nn.functional as F

def new_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    # NOTE: hidden_states can have either 1D or 2D shape.
    layer_expert_mask = getattr(self, 'current_expert_masks', None)
    #print(f"layer_expert_mask is {layer_expert_mask}")
    orig_shape = hidden_states.shape
    hidden_states = hidden_states.view(-1, self.hidden_size)
    # router_logits: (num_tokens, n_experts)
    router_logits, _ = self.gate(hidden_states)
    #print(f"router_logits is {router_logits}")
    #print(f"router_logits shape is {router_logits.shape}")
    if layer_expert_mask is not None:
        # 创建与router_logits相同shape的mask
        expert_mask = torch.zeros_like(router_logits, dtype=torch.bool)
        # 根据layer_expert_mask设置允许的专家
        expert_mask[:, layer_expert_mask] = True
        router_logits = router_logits.masked_fill(~expert_mask, float('-inf'))
    #print(f"After mask router_logits is {router_logits}")
    #print(f"After mask router_logits shape is {router_logits.shape}")
    final_hidden_states = self.experts(hidden_states, router_logits)
    return final_hidden_states.view(orig_shape)

def forward_with_expert_masks(self, *args, **kwargs):
    """
    Args:
        expert_masks: List[torch.Tensor]，每个元素对应一个MoE层的mask
    """
    #print(f"self type in forward_with_expert_masks is: {type(self)}")
    expert_masks = getattr(self, 'current_expert_masks', None)
    #print(f"forward_with_expert_masks in self.current_expert_masks is {expert_masks}")
            
    moe_layer_idx = 0
    for idx, layer in enumerate(self.model.layers):
        if hasattr(layer, 'block_sparse_moe'):
            #print(f"layer type in forward_with_expert_masks is: {type(layer)}")
            # 为每个MoE层设置对应的expert_mask
            layer.block_sparse_moe.current_expert_masks = expert_masks[moe_layer_idx] if expert_masks is not None else None
            #print(f"Moe block {idx} mask is: {layer.block_sparse_moe.current_expert_mask}")
            moe_layer_idx += 1
    
    outputs = self.original_forward(*args, **kwargs)
    return outputs

def modify_moe_layers(model):
    if hasattr(model, 'model_runner'):
        model.model_runner = model.model_runner
    #print(f"--------------------------------")
    #print(f"In Moe Model Top current_expert_masks is {model.current_expert_masks}")

    # 修改每个MoE层
    for layer in model.model.layers:
        if hasattr(layer, 'block_sparse_moe'):
            moe_layer = layer.block_sparse_moe
            moe_layer.forward = MethodType(new_forward, moe_layer)
    
    model.original_forward = model.forward
    model.forward = MethodType(forward_with_expert_masks, model)

class ExpertMaskLLM(LLM):
    """扩展LLM类以支持expert masks"""
    def __init__(self, expert_masks=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #self.current_expert_masks = None
        print(f"Using ExpertMaskLLM")
        self.llm_engine.model_executor.driver_worker.model_runner.model.current_expert_masks = expert_masks
        self.current_expert_masks = expert_masks
        print(self.llm_engine.model_executor.driver_worker.model_runner.model)
        modify_moe_layers(self.llm_engine.model_executor.driver_worker.model_runner.model)

def load_mmlu_pro():
    dataset = load_dataset("TIGER-Lab/MMLU-Pro")
    test_df, val_df = dataset["test"], dataset["validation"]
    # test_df = pd.read_parquet('./data/test-00000-of-00001.parquet')
    # val_df = pd.read_parquet('./data/validation-00000-of-00001.parquet')
    test_df = preprocess(test_df)
    val_df = preprocess(val_df)
    return test_df, val_df


def load_model(expert_masks):
    # llm = LLM(model=args.model, gpu_memory_utilization=float(args.gpu_util),
    #             tensor_parallel_size=torch.cuda.device_count(),
    #             max_model_len=max_model_length,
    #             trust_remote_code=True)
    print(f"init expert_masks is {expert_masks}")
    llm = ExpertMaskLLM(
        model=args.model, 
        gpu_memory_utilization=float(args.gpu_util),
        tensor_parallel_size=torch.cuda.device_count(),
        max_model_len=max_model_length,
        trust_remote_code=True,
        expert_masks=expert_masks)

    sampling_params = SamplingParams(temperature=0, max_tokens=max_new_tokens,
                                        stop=["Question:"])
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    return (llm, sampling_params), tokenizer


def preprocess(test_df):
    res_df = []
    for each in test_df:
        options = []
        for opt in each["options"]:
            if opt == "N/A":
                continue
            options.append(opt)
        each["options"] = options
        res_df.append(each)
    return res_df


def args_generate_path(input_args):
    scoring_method = "CoT"
    model_name = input_args.model.split("/")[-1]
    subjects = args.selected_subjects.replace(",", "-").replace(" ", "_")
    return [model_name, scoring_method, subjects]


def select_by_category(df, subject):
    res = []
    for each in df:
        if each["category"] == subject:
            res.append(each)
    return res


def format_cot_example(example, including_answer=True):
    prompt = "Question:\n"
    question = example["question"]
    options = example["options"]
    prompt += question + "\n"
    prompt += "Options:\n"
    for i, opt in enumerate(options):
        prompt += "{}. {}\n".format(choices[i], opt)
    if including_answer:
        cot_content = example["cot_content"].replace("A: Let's think step by step.",
                                                     "Answer: Let's think step by step.")
        prompt += cot_content + "\n\n"
    else:
        prompt += "Answer: Let's think step by step."
    return prompt


def generate_cot_prompt(val_df, curr, k):
    prompt = ""
    with open(f"cot_prompt_lib/initial_prompt.txt", "r") as fi:
        for line in fi.readlines():
            prompt += line
    subject = curr["category"]
    val_df = select_by_category(val_df, subject)
    val_df = val_df[: k]
    prompt = prompt.replace("{$}", subject) + "\n"
    for example in val_df:
        prompt += format_cot_example(example, including_answer=True)
    prompt += format_cot_example(curr, including_answer=False)
    return prompt


def extract_answer(text):
    pattern = r"answer is \(?([A-J])\)?"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        print("1st answer extract failed\n" + text)
        return extract_again(text)


def extract_again(text):
    match = re.search(r'.*[aA]nswer:\s*([A-J])', text)
    if match:
        return match.group(1)
    else:
        return extract_final(text)


def extract_final(text):
    pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(0)
    else:
        return None


def batch_inference(llm, sampling_params, inference_batch):
    start = time.time()
    #outputs = llm.generate(inference_batch, sampling_params)
    outputs = llm.generate(inference_batch, sampling_params)
    logging.info(str(len(inference_batch)) + "size batch costing time: " + str(time.time() - start))
    response_batch = []
    pred_batch = []
    for output in outputs:
        generated_text = output.outputs[0].text
        response_batch.append(generated_text)
        pred = extract_answer(generated_text)
        pred_batch.append(pred)
    return pred_batch, response_batch


def save_res(res, output_path):
    accu, corr, wrong = 0.0, 0.0, 0.0
    with open(output_path, "w") as fo:
        fo.write(json.dumps(res))
    for each in res:
        if not each["pred"]:
            random.seed(12345)
            x = random.randint(0, len(each["options"]) - 1)
            if x == each["answer_index"]:
                corr += 1
                # print("random hit.")
            else:
                wrong += 1
        elif each["pred"] == each["answer"]:
            corr += 1
        else:
            wrong += 1
    if corr + wrong == 0:
        return 0.0, 0.0, 0.0
    accu = corr / (corr + wrong)
    return accu, corr, wrong


@torch.no_grad()
def eval_cot(subject, model, tokenizer, val_df, test_df, output_path):
    llm, sampling_params = model
    global choices
    logging.info("evaluating " + subject)
    inference_batches = []

    for i in tqdm(range(len(test_df))):
        k = args.ntrain
        curr = test_df[i]
        prompt_length_ok = False
        prompt = None
        while not prompt_length_ok:
            prompt = generate_cot_prompt(val_df, curr, k)
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {key: value.cuda() for key, value in inputs.items()}
            length = len(inputs["input_ids"][0])
            if length < max_model_length - max_new_tokens:
                prompt_length_ok = True
            k -= 1
        inference_batches.append(prompt)

    pred_batch, response_batch = batch_inference(llm, sampling_params, inference_batches)
    res = []
    for j, curr in enumerate(test_df):
        curr["pred"] = pred_batch[j]
        curr["model_outputs"] = response_batch[j]
        res.append(curr)
    accu, corr, wrong = save_res(res, output_path)
    logging.info("this batch accu is: {}, corr: {}, wrong: {}\n".format(str(accu), str(corr), str(wrong)))

    accu, corr, wrong = save_res(res, output_path)
    return accu, corr, wrong

def load_expert_masks(moe_path):
    expert_mask= torch.load(moe_path)
    num_layers = len(expert_mask)
    expert_masks = []
    for layer_idx in range(num_layers):
        active_experts = torch.nonzero(expert_mask[layer_idx]).squeeze()
        expert_masks.append(active_experts)
    return expert_masks

def main():
    global expert_masks
    expert_masks = load_expert_masks(args.moe_path)
    print(f"after load expert_masks is {expert_masks}")
    model, tokenizer = load_model(expert_masks)
    llm, sampling_params = model
    if not os.path.exists(save_result_dir):
        os.makedirs(save_result_dir)

    full_test_df, full_val_df = load_mmlu_pro()
    all_subjects = []
    for each in full_test_df:
        if each["category"] not in all_subjects:
            all_subjects.append(each["category"])
    if args.selected_subjects == "all":
        selected_subjects = all_subjects
    else:
        selected_subjects = []
        args_selected = args.selected_subjects.split(",")
        for sub in all_subjects:
            for each in args_selected:
                if each.replace(" ", "_") in sub.replace(" ", "_"):
                    selected_subjects.append(sub)
    logging.info("selected subjects:\n" + "\n".join(selected_subjects))
    print("selected subjects:\n" + "\n".join(selected_subjects))
    sta_dict = {}
    selected_subjects = sorted(selected_subjects)
    with open(os.path.join(summary_path), 'a') as f:
        f.write("\n------category level sta------\n")
    for subject in selected_subjects:
        if subject not in sta_dict:
            sta_dict[subject] = {"corr": 0.0, "wrong": 0.0, "accu": 0.0}
        test_df = select_by_category(full_test_df, subject)
        val_df = select_by_category(full_val_df, subject)
        output_path = os.path.join(save_result_dir, "{}.json".format(subject))
        acc, corr_count, wrong_count = eval_cot(subject, model, tokenizer, val_df, test_df, output_path)
        sta_dict[subject]["corr"] = corr_count
        sta_dict[subject]["wrong"] = wrong_count
        sta_dict[subject]["accu"] = acc
        with open(os.path.join(summary_path), 'a') as f:
            f.write("Average accuracy {:.4f} - {}\n".format(sta_dict[subject]["accu"], subject))
    total_corr, total_wrong = 0.0, 0.0
    for k, v in sta_dict.items():
        total_corr += v["corr"]
        total_wrong += v["wrong"]
    total_accu = total_corr / (total_corr + total_wrong + 0.000001)
    sta_dict["total"] = {"corr": total_corr, "wrong": total_wrong, "accu": total_accu}

    #Save results
    #mask, domain, Accuracy
    mask = args.moe_path.split('/')[-1].replace('.pth', '')
    domain = args.selected_subjects

    csv_file = './results/results_phi_task2.csv'
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
    else:
        df = pd.DataFrame(columns=['mask', 'domain', 'accuracy'])
    
    new_record = pd.DataFrame({
        'mask': [mask],
        'domain': [domain], 
        'accuracy': [total_accu]
    })
    
    df = pd.concat([df, new_record], ignore_index=True)
    
    df.to_csv(csv_file, index=False)

    with open(os.path.join(summary_path), 'a') as f:
        f.write("\n------average acc sta------\n")
        weighted_acc = total_accu
        f.write("Average accuracy: {:.4f}\n".format(weighted_acc))
    with open(global_record_file, 'a', newline='') as file:
        writer = csv.writer(file)
        record = args_generate_path(args) + [time_str, weighted_acc]
        writer.writerow(record)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--selected_subjects", "-sub", type=str, default="all")
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--global_record_file", "-grf", type=str,
                        default="eval_record_collection.csv")
    parser.add_argument("--gpu_util", "-gu", type=str, default="0.8")
    parser.add_argument("--model", "-m", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument('--moe_path', type=str, required=True,
                        help='MOE专家mask文件路径 (.pth)')

    args = parser.parse_args()

    moe_path = args.moe_path
    print("--------------------------------")
    print(f"select moe is {moe_path}")
    print("--------------------------------")

    os.makedirs(args.save_dir, exist_ok=True)
    global_record_file = args.global_record_file
    save_result_dir = os.path.join(
        args.save_dir, "/".join(args_generate_path(args))
    )
    file_prefix = "-".join(args_generate_path(args))
    timestamp = time.time()
    time_str = time.strftime('%m-%d_%H-%M', time.localtime(timestamp))
    file_name = f"{file_prefix}_{time_str}_summary.txt"
    summary_path = os.path.join(args.save_dir, "summary", file_name)
    os.makedirs(os.path.join(args.save_dir, "summary"), exist_ok=True)
    os.makedirs(save_result_dir, exist_ok=True)
    save_log_dir = os.path.join(args.save_dir, "log")
    os.makedirs(save_log_dir, exist_ok=True)
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.FileHandler(os.path.join(save_log_dir,
                                                                   file_name.replace("_summary.txt",
                                                                                     "_logfile.log"))),
                                  logging.StreamHandler(sys.stdout)])

    main()


