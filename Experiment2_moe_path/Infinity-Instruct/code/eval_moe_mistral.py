import torch  
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, set_seed
from transformers import AutoTokenizer 
from transformers import BitsAndBytesConfig
import torch
from tqdm import tqdm  
import random
import numpy as np
import pandas as pd
import os
import argparse
os.environ["CUDA_VISIBLE_DEVICES"]='0'


def set_random_seed(seed):
    if seed is not None:
        set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
set_random_seed(42)


import torch
from types import MethodType
from typing import Optional
import torch.nn.functional as F

def new_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """
    Args:
        hidden_states: 输入张量
        layer_expert_mask: 当前层的专家mask，shape为[num_experts]
    """
    layer_expert_mask = getattr(self, 'current_expert_mask', None)
    #print(f"layer_expert_mask is {layer_expert_mask}")
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    if self.training and self.jitter_noise > 0:
        hidden_states *= torch.empty_like(hidden_states).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)
    hidden_states = hidden_states.view(-1, hidden_dim)

    router_logits = self.gate(hidden_states)
    
    if layer_expert_mask is not None:
        # 创建与router_logits相同shape的mask
        expert_mask = torch.zeros_like(router_logits, dtype=torch.bool)
        # 根据layer_expert_mask设置允许的专家
        expert_mask[:, layer_expert_mask] = True
        router_logits = router_logits.masked_fill(~expert_mask, float('-inf'))
    
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    #print(f"routing_weights is {routing_weights}")
    routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
    #print(f"selected_experts is {selected_experts}")
    routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    # we cast back to the input dtype
    routing_weights = routing_weights.to(hidden_states.dtype)

    final_hidden_states = torch.zeros(
        (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
    )
    
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
    #[batch_size * sequence_length, top_k, num_experts] --> [num_experts, top_k, batch_size * sequence_length]
    # Loop over all available experts in the model and perform the computation on each expert
    for expert_idx in range(self.num_experts):
        expert_layer = self.experts[expert_idx]
        idx, top_x = torch.where(expert_mask[expert_idx])

        # Index the correct hidden states and compute the expert hidden state for
        # the current expert. We need to make sure to multiply the output hidden
        # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
        current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
        current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

        # However `index_add_` only support torch tensors for indexing so we'll use
        # the `top_x` tensor here.
        final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
    return final_hidden_states, router_logits

def modify_moe_layers(model):
    # 修改每个MoE层
    for layer in model.model.layers:
        if hasattr(layer, 'block_sparse_moe'):
            moe_layer = layer.block_sparse_moe
            moe_layer.forward = MethodType(new_forward, moe_layer)
    
    model.original_forward = model.forward
    model.forward = MethodType(forward_with_expert_masks, model)



def forward_with_expert_masks(self, *args, expert_masks=None, **kwargs):
    """
    Args:
        expert_masks: List[torch.Tensor]，每个元素对应一个MoE层的mask
    """
    moe_layer_idx = 0
    for idx, layer in enumerate(self.model.layers):
        if hasattr(layer, 'block_sparse_moe'):
            # 为每个MoE层设置对应的expert_mask
            layer.block_sparse_moe.current_expert_mask = expert_masks[moe_layer_idx] if expert_masks is not None else None
            #print(f"Moe block {idx} mask is: {layer.block_sparse_moe.current_expert_mask}")
            moe_layer_idx += 1
    
    outputs = self.original_forward(*args, **kwargs)
    return outputs


if __name__ == "__main__":
    # quantization_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_compute_dtype=torch.float16,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_use_double_quant=True
    # )
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='计算模型困惑度')
    parser.add_argument('--model_path', type=str, required=True,
                        help='模型路径')
    parser.add_argument('--data_path', type=str, required=True,
                        help='数据文件路径 (.parquet)')
    parser.add_argument('--moe_path', type=str, required=True,
                        help='MOE专家mask文件路径 (.pth)')
    parser.add_argument('--output_prefix', type=str, default='select_moe_model_outputs',
                        help='输出文件前缀')
    parser.add_argument('--batch_size', type=int, default=26,
                        help='批处理大小')
    
    args = parser.parse_args()
    
    # 使用参数替换原来的硬编码值
    ori_model_path = args.model_path
    data_path = args.data_path
    moe_path = args.moe_path
    BATCH_SIZE = args.batch_size
    print("--------------------------------")
    print(f"data is {data_path}")
    print(f"select moe is {moe_path}")
    print("--------------------------------")

    config = AutoConfig.from_pretrained(ori_model_path, trust_remote_code=True)
    bc_model = AutoModelForCausalLM.from_pretrained(
        ori_model_path,
        config=config,
        torch_dtype=torch.float16,
        #low_cpu_mem_usage=False,
        trust_remote_code=True,
        revision='main',
        device_map='auto',
    # quantization_config=quantization_config,
        #use_flash_attention_2=True
    )
    tokenizer = AutoTokenizer.from_pretrained(ori_model_path) 
    tokenizer.pad_token = tokenizer.eos_token

    ## 加载数据集
    # from datasets import load_dataset
    # dataset = load_dataset("/root/Infinity-Instruct/0625", "default")

    ##筛选符合要求的文本，标签分别为'财务、金融与商业知识'和'医学、药学与健康知识'
    # cnt = 0
    # data_inputs = []
    # target = []
    # label_for_inputs = []

    # for i in tqdm(range(len(dataset['train'])-5000, len(dataset['train']))):
    #     if ('财务、金融与商业知识' in dataset['train'][i]['label']['cate_ability_zh'])^('医学、药学与健康知识' in dataset['train'][i]['label']['cate_ability_zh']):
    #         data_inputs.append(dataset['train'][i]['conversations'][0]['value'])
    #         target.append(dataset['train'][i]['conversations'][1]['value'])
    #         label_for_inputs.append(set(dataset['train'][i]['label']['cate_ability_zh']) & set(['财务、金融与商业知识', '医学、药学与健康知识']))
    data = pd.read_parquet(data_path)
    #data = data.loc[6001:,]
    # data = data[data.perplexity.isna()].reset_index(drop=True)
    data_inputs = data['input_text'].tolist()
    target = data['target_text'].tolist()
    label_for_inputs = data['label'].tolist()

    # 根据data_inputs中文本长度排序所有列表
    indices = sorted(range(len(data_inputs)), key=lambda k: len(data_inputs[k].split(" ")))
    data_inputs = [data_inputs[i] for i in indices]
    target = [target[i] for i in indices]
    label_for_inputs = [label_for_inputs[i] for i in indices]

    #bc_model.original_forward = bc_model.forward
    #bc_model.forward = MethodType(forward_with_expert_masks, bc_model)
    modify_moe_layers(bc_model)

    expert_mask= torch.load(moe_path)
    num_layers = len([layer for layer in bc_model.model.layers if hasattr(layer, 'block_sparse_moe')])
    expert_masks = []
    for layer_idx in range(num_layers):
        active_experts = torch.nonzero(expert_mask[layer_idx]).squeeze()
        expert_masks.append(active_experts)

    all_results = []
    abnormal_results = []  # 新增：存储异常数据
    contextual_perplexity = 0
    total_loss = 0
    valid_samples = 0  # 新增：记录有效样本数量

    # 修改为真正的批处理方式
    for i in tqdm(range(0, len(data_inputs), BATCH_SIZE)):
        batch_inputs = data_inputs[i:i+BATCH_SIZE]
        batch_targets = target[i:i+BATCH_SIZE]
        batch_labels = label_for_inputs[i:i+BATCH_SIZE]
        
        # 分别对输入和目标进行截断，确保两者都有足够的长度
        max_input_length = 256  # 为输入文本预留的最大长度
        max_target_length = 256  # 为目标文本预留的最大长度
        
        # 分别处理输入和目标
        tokenized_inputs = tokenizer(batch_inputs, 
                                max_length=max_input_length,
                                truncation=True,
                                return_tensors="pt",
                                padding=True)
        
        tokenized_targets = tokenizer(batch_targets,
                                    max_length=max_target_length,
                                    truncation=True,
                                    return_tensors="pt",
                                    padding=True)
        
        # 合并输入和目标的token ids
        combined_input_ids = []
        attention_mask = []
        for inp_ids, tgt_ids, inp_mask, tgt_mask in zip(
            tokenized_inputs['input_ids'],
            tokenized_targets['input_ids'],
            tokenized_inputs['attention_mask'],
            tokenized_targets['attention_mask']
        ):
            # 移除目标开头的特殊token（如果有的话）
            if tgt_ids[0] == tokenizer.bos_token_id:
                tgt_ids = tgt_ids[1:]
                tgt_mask = tgt_mask[1:]
                
            combined_ids = torch.cat([inp_ids, tgt_ids])
            combined_mask = torch.cat([inp_mask, tgt_mask])
            
            combined_input_ids.append(combined_ids)
            attention_mask.append(combined_mask)
        
        # 将列表转换为张量并进行padding
        combined_input_ids = torch.nn.utils.rnn.pad_sequence(combined_input_ids, batch_first=True)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True)
        
        inputs = {
            'input_ids': combined_input_ids.to(bc_model.device),
            'attention_mask': attention_mask.to(bc_model.device)
        }
        
        # 获取每个样本的input长度，用于确定target的起始位置
        input_lengths = [len(ids) for ids in tokenized_inputs['input_ids']]
        
        with torch.no_grad():
            outputs= bc_model(**inputs, expert_masks=expert_masks)
            batch_perplexities = []
            
            # 对批次中的每个样本计算perplexity
            for idx, target_start_idx in enumerate(input_lengths):
                logits = outputs.logits[idx:idx+1, target_start_idx-1:-1, :]
                #target_ids = inputs["input_ids"][idx:idx+1, target_start_idx:]

                # 获取目标序列的id和attention_mask
                target_ids = inputs["input_ids"][idx:idx+1, target_start_idx:]
                target_attention_mask = attention_mask[idx:idx+1, target_start_idx:]
                # 创建标签，并将填充位置设置为-100
                labels = target_ids.clone()
                labels[target_attention_mask == 0] = -100

                assert logits.size(1) == labels.size(1)
                
                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
                target_loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
                
                perplexity = torch.exp(target_loss).item()
                batch_perplexities.append(perplexity)
                print(f"this round: {perplexity}")
                
                # 修改：只将正常的困惑度值加入总和
                if not (torch.isinf(torch.tensor(perplexity)) or torch.isnan(torch.tensor(perplexity))):
                    contextual_perplexity += perplexity
                    valid_samples += 1

        # 保存每个batch的结果
        for j, perplexity in enumerate(batch_perplexities):
            result = {
                'input_text': batch_inputs[j],
                'target_text': batch_targets[j],
                'label': list(batch_labels[j]),
                'perplexity': perplexity
            }
            
            # 区分正常和异常数据
            if torch.isinf(torch.tensor(perplexity)) or torch.isnan(torch.tensor(perplexity)):
                abnormal_results.append(result)
            else:
                all_results.append(result)

    # 计算平均困惑度时使用有效样本数
    if valid_samples > 0:
        contextual_perplexity /= valid_samples
    print(f"Contextual Perplexity (excluding inf/nan): {contextual_perplexity}")
    print(f"Number of abnormal samples: {len(abnormal_results)}")

    # 分别保存正常和异常数据
    df_normal = pd.DataFrame(all_results)
    df_abnormal = pd.DataFrame(abnormal_results)

    threshold = df_normal['perplexity'].quantile(0.95) 
    filtered_df = df_normal[df_normal['perplexity'] <= threshold] 

    #Save results
    # mask, domain, mean_perplexity, median_perplexity
    mean = df_normal.perplexity.mean()
    median = df_normal.perplexity.median()

    mask = args.moe_path.split('/')[-1].replace('.pth', '')
    domain = args.data_path.split('/')[-1].replace('.parquet', '')

    csv_file = './results/results_mistral_moe.csv'
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
    else:
        df = pd.DataFrame(columns=['mask', 'domain', 'mean_perplexity', "median_perplexity"])
    
    new_record = pd.DataFrame({
        'mask': [mask],
        'domain': [domain], 
        'mean_perplexity': [mean],
        "median_perplexity": [median]
    })
    
    df = pd.concat([df, new_record], ignore_index=True)
    
    df.to_csv(csv_file, index=False)

    # 使用参数中的output_prefix构建输出文件名
    df_normal.to_parquet(f'./output/mistral_{domain}_{args.output_prefix}_normal.parquet')
    df_abnormal.to_parquet(f'./output/mistral_{domain}_{args.output_prefix}_abnormal.parquet')


