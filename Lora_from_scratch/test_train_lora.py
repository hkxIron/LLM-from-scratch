import os
import copy
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
from typing import List
from einops import rearrange
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoModelForCausalLM
from typing import Dict, List, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments
from transformers.models.llama.tokenization_llama_fast import LlamaTokenizer
import argparse

from Lora_from_scratch.train_lora import chatglm_inference, test_inference_stream
from lora_layer import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 设置 device，能用 cuda 就用 cuda，苹果 M 系列可以用 mps
dtype = torch.bfloat16 if device != 'cpu' and torch.cuda.is_bf16_supported() else torch.float32
print(f'device: {device}\ndtype: {dtype}')

def test_lora_load_unload():
    config = AutoConfig.for_model('llama')
    config.hidden_size = 24
    config.intermediate_size = config.hidden_size * 4
    config.num_attention_heads = 4
    config.num_hidden_layers = 4
    config.num_key_value_heads = 2 # group attention, 两个head组成一个group
    config.vocab_size = 128

    # 未使用lora替换
    raw_model = AutoModel.from_config(config)
    # trainable params: 37,848 || all params: 37,848 || trainable%: 100.0000
    print_trainable_parameters(raw_model)
    #tokenizer = AutoTokenizer.from_pretrained('/home/hkx/data/work/hf_data_and_model/models/NousResearch/Llama-2-7b-hf')

    # 使用lora替换后
    lora_model = copy.deepcopy(raw_model)
    replace_linear_with_lora(lora_model, rank=8, alpha=16)
    # trainable params: 16,896 || all params: 54,744 || trainable%: 30.8637
    print_trainable_parameters(lora_model)

    # 与使用huggingface的lora相比
    from peft import LoraConfig, get_peft_model
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules='all-linear',  # 太低版本的 peft 不支持这种做法
    )
    peft_lora_model = copy.deepcopy(raw_model)
    peft_lora_model = get_peft_model(peft_lora_model, lora_config)
    print("hf peft lora")
    peft_lora_model.print_trainable_parameters() # 结果与自己实现的一模一样
    # trainable params: 16,896 || all params: 54,744 || trainable%: 30.8637

    """
    下面测试lora的unload与load
    """
    # 创建一个测试 tensor
    batch_size = 2
    seq_len = 8
    test_input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    # 开测试模式，让 BA 非零
    lora_model = copy.deepcopy(raw_model)
    replace_linear_with_lora(lora_model, test_mode=True)

    # 原模型的前向结果
    raw_model.eval()
    print_trainable_parameters(raw_model)  # 检查参数和可训练情况
    raw_res = raw_model(test_input_ids).last_hidden_state
    print(raw_res.shape)
    print(raw_res[:,:,0])

    # 第一次直接初始化 lora 的前向结果
    lora_model.eval()
    print_trainable_parameters(lora_model)  # 检查参数和可训练情况
    before_unload_res = lora_model(test_input_ids).last_hidden_state
    print(before_unload_res.shape)
    print(before_unload_res[:,:,0])

    # 卸载 lora 后的前向结果
    lora_param = unload_lora_then_save(lora_model, adapter_file_name="lora_adapter.pt")
    lora_key_value = get_simple_lora_param_desc(lora_param)
    print(lora_key_value)
    lora_model.eval()
    print_trainable_parameters(lora_model)  # 检查参数和可训练情况
    unload_res = lora_model(test_input_ids).last_hidden_state
    print(unload_res.shape)
    print(unload_res[:,:,0])

    # 重新装载 lora 后的前向结果
    load_and_merge_lora_from_file(lora_model, adapter_file_name="lora_adapter.pt")
    lora_model.eval()
    print_trainable_parameters(lora_model)  # 检查参数和可训练情况
    load_res = lora_model(test_input_ids).last_hidden_state
    print(load_res.shape)
    print(load_res[:,:,0])

    # 单独保存lora权重
    only_lora_param = save_lora_model(lora_model, adapter_file_name="lora_adapter2.pt")
    only_lora_key_value = get_simple_lora_param_desc(only_lora_param)
    print(only_lora_key_value)
    assert only_lora_key_value == lora_key_value


    print(torch.allclose(raw_res, unload_res, atol=1e-6))  # 应为 True
    print(torch.allclose(before_unload_res, load_res, atol=1e-6))  # 应为 True
    print(torch.allclose(raw_res, load_res, atol=1e-6))  # 应为 False

def test_inference1():
    base_path="/media/hkx/win/hkx/ubuntu/work/hf_data_and_model/"
        # 下面这个生物数据集要小很多
    model_name_or_path = f'{base_path}/models/Qwen1.5-0.5B/'
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path) # 使用的是GPT2Tokenizer,不是LlamaTokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=dtype, device_map='auto').to(device)

    model.eval()

    for test_text in [
        'Describe the process of bacterial conjugation and its significance in the context of antibiotic resistance.',
        'Explain the role of insulin in the body and how insulin resistance affects blood sugar levels.',
        'Provide recommendations for lifestyle changes that can help improve the overall health of a patient with type 2 diabetes.',
    ]:
        print('=' * 80)
        last_text = ''
        for text in chatglm_inference(model, tokenizer, test_text, streaming=True):
            # 在streaming模式下
            # text是完整的文本，需要去掉前面的部分
            # last_text="Bacterial conjugation is"
            # cur_text = "Bacterial conjugation is a"
            cur_text = text.replace(last_text, '')
            print(cur_text, end='', flush=True)
            last_text = text
        print('\n')

def load_and_infer():
    from datasets import config
    print(f'huggingface cache path:{config.HF_DATASETS_CACHE}')

    dtype = torch.bfloat16 if device != 'cpu' and torch.cuda.is_bf16_supported() else torch.float32  # 使用float16很快出现nan,而使用float32则不会
    # dtype = torch.bfloat16 if device != 'cpu' and torch.cuda.is_bf16_supported() else torch.float16
    print(f'device: {device}\ndtype: {dtype}')
    from datasets import config
    print(f'huggingface cache path:{config.HF_DATASETS_CACHE}')

    use_chatglm = True
    base_path = "/media/hkx/win/hkx/ubuntu/work/hf_data_and_model/"
    if use_chatglm:
        # 下面这个生物数据集要小很多
        model_name_or_path = f'{base_path}/models/Qwen1.5-0.5B/'
        data_name_or_path = f'{base_path}/datas/bioinstruct/'  # sft格式数据集， {“input”, "instruction", "output"}
    else:
        model_name_or_path = f'{base_path}/models/ahxt/LiteLlama-460M-1T'
        data_name_or_path = f'{base_path}/datas/vicgalle/alpaca-gpt4'  # sft 格式的数据， {“input”, "instruction", "output", "text"}, "text"是将instruction+input+output组合而成

    # tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)  # 使用的是GPT2Tokenizer,不是LlamaTokenizer
    tokenizer.pad_token_id = tokenizer.eos_token_id  # pad_token = eos_token
    tokenizer.padding_side = 'left'
    # Transformers 开箱即用地支持简单的流水线并行。为此，只需使用 device_map="auto"加载模型，它就会自动将不同层放到相应的 GPU 上，使模型的加载和推理更加灵活和高效()
    # load_in_8bit = True,需要GPU
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=dtype, device_map='auto').to(device)
    model_no_lora = copy.deepcopy(model)
    test_text_list = ['Describe the process of bacterial conjugation and its significance in the context of antibiotic resistance.']
    print("no lora")
    # ------------
    test_inference_stream(model_no_lora, tokenizer, test_text_list)
    lora_file = "only_lora.pt"
    load_and_merge_lora_from_file(model, lora_file)
    # ------------
    print("with lora")
    test_inference_stream(model, tokenizer, test_text_list)
    """
    no lora
    ================================================================================
    We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)
    <|im_start|>system
    You are a helpful assistant<|im_end|>
    <|im_start|>user
    Describe the process of bacterial conjugation and its significance in the context of antibiotic resistance.<|im_end|>
    <|im_start|>assistant
    Bacterial conjugation is a process in which a bacterium transfers its genetic material (DNA) to another bacterium, either through the formation of a conjugative plasmid or by the use of a plasmid that contains the bacterial chromosome. This process is crucial in the development of antibiotic resistance in bacteria.

    The process of bacterial conjugation involves the following steps:

    1. Transformation: The bacterium infects a host cell, either through direct contact or by infecting a host cell that is already infected. The bacterium then replicates its DNA into a plasmid, which is then transferred to the host cell.

    2. Plasmid formation: The bacterium creates a plasmid by synthesizing DNA from its own DNA. The plasmid is then packaged into a small

    with lora
    ================================================================================
    <|im_start|>system
    You are a helpful assistant<|im_end|>
    <|im_start|>user
    Describe the process of bacterial conjugation and its significance in the context of antibiotic resistance.<|im_end|>
    <|im_start|>assistant
    Bacterial conjugation is a process in which a bacterium is able to produce an antibiotic-like compound in its cell membrane, thereby increasing its resistance to the antibiotic. This process is essential inuates the resistance to antibiotics, as the antibiotic is no longer effective against the bacterium.
    """

if __name__ == '__main__':
    load_and_infer()
    #test_lora_load_unload()
    #test_inference1()
