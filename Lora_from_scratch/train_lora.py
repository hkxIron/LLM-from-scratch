import os
import copy
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
from typing import List
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoModelForCausalLM
from typing import Dict, List, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments
from transformers.models.llama.tokenization_llama_fast import LlamaTokenizer
import argparse
from lora_layer import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 设置 device，能用 cuda 就用 cuda，苹果 M 系列可以用 mps
use_chatglm = True


def collate_fn(batch: List[Dict[str, Any]], tokenizer)->Dict[str, torch.LongTensor]:
    """
        将List[Dict[str, Any]] 进行padding后转成Dict[str, Tensor]
        1. 将batch list中的多条样本变为一个batch中的单条样本
        2. 对一个batch中的样本进行padding
    """
    max_len_in_batch = max(len(item['input_ids']) for item in batch) # 由于在convert_text_to_id中已经进行了max_len截断,此处不再需要考虑max_len

    input_ids = []
    attention_mask = []
    labels = []

    # 同一个batch内的数据，padding补齐到最大值
    for item in batch:
        input_id = item['input_ids']
        attention_mask_item = item['attention_mask']
        label = item['labels']

        # 计算填充长度
        pad_len = max_len_in_batch - len(input_id)

        # 左填充
        input_ids.append([tokenizer.eos_token_id] * pad_len + input_id) # input_ids padding填充eos
        attention_mask.append([0] * pad_len + attention_mask_item) # attention_mask padding填充0
        labels.append([-100] * pad_len + label) # labels padding也填充-100,不过具体是多少并不重要，因为还有attention_mask会将0的位置loss都屏蔽掉

    # 将列表转换为张量
    input_ids = torch.LongTensor(input_ids)
    attention_mask = torch.LongTensor(attention_mask)
    labels = torch.LongTensor(labels)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
    }

# 定义训练数据集
class SFTDataset(Dataset):
    def __init__(self,
        tokenizer: AutoTokenizer,
        data_path: str,
        load_local: bool = False,
        max_len: int = 256,
        split_len: str = '1%',
    ):
        super().__init__()
        self.tokenizer = tokenizer

        if load_local: # 加载本地数据
            self.ds = load_dataset('json', data_dir=data_path, split=f'train[:{split_len}]')
        else:
            self.ds = load_dataset(data_path, split=f'train[:{split_len}]')
        self.max_len = max_len

        def convert_text_to_ids(example:Dict[str, str]):
            # 提取 instruction 和 input
            instruction = example['instruction'].strip()
            input = example['input'].strip()
            output = example['output'].strip()

            # 构造模板
            instruction_prompt = f"Human: {instruction}\n" + \
                                    (f"{input}\n" if len(input) > 0 else "") + \
                                    "Assistant: "
            output_prompt = f"{output}\n"

            # 截断，最大不超过 max_len
            tokenized_instruction = self.tokenizer(instruction_prompt, add_special_tokens=False)['input_ids'] # 注意：不加特殊token,如 Bos, eos
            tokenized_output = self.tokenizer(output_prompt, add_special_tokens=False)['input_ids']
            # 将instruction 和 output 直接拼接起来,注意中间没有任何special token
            tokenized_prompt = (tokenized_instruction + tokenized_output)[:self.max_len] # 最大长度截断

            # 构造 input_ids, attention_mask, labels
            input_ids = tokenized_prompt[:-1] # 由于是自己计算loss，所以input_id只取了0~t-2的token
            labels = tokenized_prompt[1:] # labels取了1~t-1的token
            # input部分不计算loss设为0, output部分需要计算loss设为1
            padding_mask = ([0] * len(tokenized_instruction) + [1] * (len(tokenized_output)))[:self.max_len][1:]

            return {
                'input_ids': torch.LongTensor(input_ids),
                'attention_mask': torch.LongTensor(padding_mask),
                'labels': torch.LongTensor(labels),
            }

        self.ds = self.ds.map(convert_text_to_ids,
                              batched=False,
                              remove_columns=self.ds.column_names,
                              desc='Processing dataset', )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index: int):
        return self.ds[index]


"""
# 定义训练数据集
这里的构造的数据是按照 ChatML 格式构造的，即：
由于 Qwen 本身用的也是 ChatML 格式，所以 tokenizer 已经实现了 apply_chat_template 方法，可以直接用。
"""
class ChatGLMSFTDataset(Dataset):
    def __init__(self,
        tokenizer: AutoTokenizer,
        data_path: str,
        load_local: bool = False,
        max_len: int = 256,
        split_len: str = '1%',
    ):
        super().__init__()
        self.tokenizer = tokenizer

        if load_local:
            ds = load_dataset('json', data_dir=data_path, split=f'train[:{split_len}]')
        else:
            ds = load_dataset(data_path, split=f'train[:{split_len}]')
        self.max_len = max_len

        def convert_text_to_ids(example:Dict[str, str]):
            # 提取 instruction 和 input
            instruction = example['instruction'].strip()
            input = example['input'].strip()
            output = example['output'].strip()

            """
            数据格式如下：
            <|im_start|>system
            You are a helpful assistant<|im_end|>
            <|im_start|>user
            {指令}<|im_end|>
            <|im_start|>assistant
            {回复}<|im_end|>
            """
            # 构造模板
            instruction_msg = [
                {"role": "user", "content": (instruction + f"\n{input}") if len(input) > 0 else instruction}
            ]
            tokenized_instruction = tokenizer.apply_chat_template(instruction_msg, tokenize=True, add_generation_prompt=True)
            tokenized_output = tokenizer(output + "<|im_end|>" + f"{tokenizer.eos_token}\n")['input_ids']

            # 截断，最大不超过 max_len
            tokenized_prompt = (tokenized_instruction + tokenized_output)[:self.max_len]

            # 构造 input_ids, attention_mask, labels
            input_ids = tokenized_prompt[:-1] # 由于是自己计算loss，所以input_id只取了0~t-2的token
            labels = tokenized_prompt[1:] # labels取了1~t-1的token
            # input部分不计算loss设为0, output部分需要计算loss设为1
            padding_mask = ([0] * len(tokenized_instruction) + [1] * (len(tokenized_output)))[:self.max_len][1:] # 注意：padding_mask向右移一位

            return {
                'input_ids': input_ids,
                'attention_mask': padding_mask,
                'labels': labels,
                'all_input_ids': tokenized_prompt
            }

        self.ds = ds.map(
            convert_text_to_ids,
            batched=False,
            remove_columns=ds.column_names, # 原始的列名全部删除，包括：instruction, input, output
            desc='Processing dataset',
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index: int):
        return self.ds[index]

def inference(
    model,
    tokenizer,
    text: str,
    max_new_tokens: int = 200,
    do_sample: bool = True,
    top_k: int = 40,
    temperature: float = 0.3,
):
    instruction_prompt = f"Human: {text}\nAssistant: "
    prompt = tokenizer(instruction_prompt, return_tensors='pt', add_special_tokens=False).to(device)
    outputs = model.generate(
        **prompt,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        top_k=top_k,
        temperature=temperature,
    )
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return response


"""
下面的推理方法用了流式输出，具体实现我们会在第三篇文章中解读
"""
def chatglm_inference(
        model,
        tokenizer,
        text: str,
        max_new_tokens: int = 160,
        do_sample: bool = True,
        temperature: float = 0.3,
        print_inputs: bool = True,
        streaming: bool = False,
):
    # 构建输入，模板要和 Dataset 中一致
    prompt_msg = [
        {"role": "user", "content": text}
    ]
    prompt = tokenizer.apply_chat_template(prompt_msg, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to(device)
    input_ids = inputs['input_ids']
    im_end_id = tokenizer.encode("<|im_end|>")[0]

    # 是否打印输入部分
    if print_inputs:
        print(prompt, end='')

    # 生成
    stop_words = [tokenizer.eos_token_id, im_end_id]
    generated_tokens = []

    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(input_ids)
        # logits:[batch, seq_len, vocab_size]
        logits = outputs.logits[:, -1, :]

        # 不同采样方式
        if do_sample:
            logits = logits / temperature # 温度越小，马太效应越强，稳定性越强
            probs = F.softmax(logits, dim=-1)
            # next_token:[batch, 1, 1]
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            # 贪婪解码
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

        # 遇到停止符则终止
        if next_token.item() in stop_words:
            break

        generated_tokens.append(next_token.item())

        # 流式输出
        if streaming:
            yield tokenizer.decode(generated_tokens)

        # 更新输入
        # input_ids: [batch, seq_len]
        input_ids = torch.cat([input_ids, next_token], dim=-1)

    generated_text = tokenizer.decode(generated_tokens)
    return generated_text

def train_lora(args):
    from datasets import config
    print(f'huggingface cache path:{config.HF_DATASETS_CACHE}')

    dtype = torch.bfloat16 if device != 'cpu' and torch.cuda.is_bf16_supported() else torch.float32 # 使用float16很快出现nan,而使用float32则不会
    #dtype = torch.bfloat16 if device != 'cpu' and torch.cuda.is_bf16_supported() else torch.float16
    print(f'device: {device}\ndtype: {dtype}')
    from datasets import config
    print(f'huggingface cache path:{config.HF_DATASETS_CACHE}')

    """
    这个模型是llama2的小型版,只有460M参数，训练的token数约为：0.98T, 这里是以1000为进制，而不是1024
    KB:1e+3(千)
    MB:1e+6(百万)
    GB:1e+9(十亿)
    TB:1e+12(万亿)
     
    KB->MB->GB->TB:存储字节数,以1024为进制
    k->M->B->T:数量,以1000为进制
    The model was trained with ~1T tokens (0.98T). num of tokens = steps * length * batch_size=499679*1024*192=98240888832≈0.98T.
    """

    base_path="/media/hkx/win/hkx/ubuntu/work/hf_data_and_model/"
    if use_chatglm:
        # 下面这个生物数据集要小很多
        model_name_or_path = f'{base_path}/models/Qwen1.5-0.5B/'
        data_name_or_path = f'{base_path}/datas/bioinstruct/' # sft格式数据集， {“input”, "instruction", "output"}
    else:
        model_name_or_path = f'{base_path}/models/ahxt/LiteLlama-460M-1T'
        data_name_or_path = f'{base_path}/datas/vicgalle/alpaca-gpt4' # sft 格式的数据， {“input”, "instruction", "output", "text"}, "text"是将instruction+input+output组合而成

    #tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path) # 使用的是GPT2Tokenizer,不是LlamaTokenizer
    tokenizer.pad_token_id = tokenizer.eos_token_id # pad_token = eos_token
    tokenizer.padding_side = 'left'
    # Transformers 开箱即用地支持简单的流水线并行。为此，只需使用 device_map="auto"加载模型，它就会自动将不同层放到相应的 GPU 上，使模型的加载和推理更加灵活和高效()
    # device_map = {
    #     "transformer.word_embeddings": 0,
    #     "transformer.word_embeddings_layernorm": 0,
    #     "lm_head": "cpu",
    #     "transformer.h": 0,
    #     "transformer.ln_f": 0,
    # }
    # load_in_8bit = True,需要GPU
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=dtype, device_map='auto').to(device)

    # 获取 lora model
    replace_linear_with_lora(model, rank=8, alpha=16, dropout_p=0.0)
    model.to(device)

    # 查看可训练参数
    print_trainable_parameters(model)

    if use_chatglm:
        ds = ChatGLMSFTDataset(tokenizer=tokenizer, data_path=data_name_or_path, load_local=False)
    else:
        ds = SFTDataset(tokenizer=tokenizer, data_path=data_name_or_path, load_local=False)

    print(len(ds[0]['input_ids']))
    print(len(ds[0]['attention_mask']))
    print(len(ds[0]['labels']))

    print(f"{tokenizer.convert_tokens_to_ids(['<|endoftext|>'])=}")
    print(f"{tokenizer.eos_token_id=}")
    print(f"{tokenizer.convert_ids_to_tokens([0])=}")
    print(f"{tokenizer.convert_ids_to_tokens([198])=}")

    print("=========")
    print(f"{ds[0]['input_ids']=}")
    print(f"{ds[0]['attention_mask']=}")
    print(f"{ds[0]['labels']=}")
    print(f"{ds[0]['all_input_ids']=}")

    print("=========")
    print(tokenizer.batch_decode(ds[0]['input_ids'], skip_special_tokens=False)) #  Whether or not to remove special tokens in the decoding
    print("=========")
    print(tokenizer.convert_ids_to_tokens(ds[0]['input_ids'], skip_special_tokens=False)) #  'Ġthe' -> ' the'
    print("=========")

    print(tokenizer.decode(ds[0]['input_ids'], skip_special_tokens=False)) #
    print("=========")
    print(ds[0]['attention_mask'])
    print("=========")
    print(tokenizer.decode(ds[0]['labels'], skip_special_tokens=False)) #
    print("=========")
    print(tokenizer.decode(ds[0]['all_input_ids'], skip_special_tokens=False)) #
    print("=========")

    """
    tokenizer.convert_tokens_to_ids(['<|endoftext|>'])=[151643]
    tokenizer.eos_token_id=151643
    tokenizer.convert_ids_to_tokens([0])=['!']
    tokenizer.convert_ids_to_tokens([198])=['Ċ']
    =========
    ds[0]['input_ids']=[151644, 8948, 198, 2610, 525, 264, 10950, 17847, 151645, 198, 151644, 872, 198, 28301, 1437, 279, 1887, 16688, 504, 279, 3897, 6457, 1895, 49465, 624, 785, 8720, 594, 6543, 1273, 3059, 8542, 458, 26163, 304, 25506, 54967, 11, 11689, 49412, 323, 22465, 11, 892, 13230, 4650, 25506, 5557, 13, 22406, 11, 279, 8720, 594, 62759, 8542, 264, 38985, 25506, 13, 151645, 198, 151644, 77091, 198, 785, 8720, 702, 11929, 315, 25506, 5557, 323, 264, 38985, 25506, 13, 151645, 151643]
    ds[0]['attention_mask']=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ds[0]['labels']=[8948, 198, 2610, 525, 264, 10950, 17847, 151645, 198, 151644, 872, 198, 28301, 1437, 279, 1887, 16688, 504, 279, 3897, 6457, 1895, 49465, 624, 785, 8720, 594, 6543, 1273, 3059, 8542, 458, 26163, 304, 25506, 54967, 11, 11689, 49412, 323, 22465, 11, 892, 13230, 4650, 25506, 5557, 13, 22406, 11, 279, 8720, 594, 62759, 8542, 264, 38985, 25506, 13, 151645, 198, 151644, 77091, 198, 785, 8720, 702, 11929, 315, 25506, 5557, 323, 264, 38985, 25506, 13, 151645, 151643, 198]
    ds[0]['all_input_ids']=[151644, 8948, 198, 2610, 525, 264, 10950, 17847, 151645, 198, 151644, 872, 198, 28301, 1437, 279, 1887, 16688, 504, 279, 3897, 6457, 1895, 49465, 624, 785, 8720, 594, 6543, 1273, 3059, 8542, 458, 26163, 304, 25506, 54967, 11, 11689, 49412, 323, 22465, 11, 892, 13230, 4650, 25506, 5557, 13, 22406, 11, 279, 8720, 594, 62759, 8542, 264, 38985, 25506, 13, 151645, 198, 151644, 77091, 198, 785, 8720, 702, 11929, 315, 25506, 5557, 323, 264, 38985, 25506, 13, 151645, 151643, 198]
    =========
    ['<|im_start|>', 'system', '\n', 'You', ' are', ' a', ' helpful', ' assistant', '<|im_end|>', '\n', '<|im_start|>', 'user', '\n', 'Ident', 'ify', ' the', ' main', ' conclusion', ' from', ' the', ' provided', ' medical', ' report', ' excerpt', '.\n', 'The', ' patient', "'s", ' blood', ' test', ' results', ' showed', ' an', ' elevation', ' in', ' liver', ' enzymes', ',', ' specifically', ' ALT', ' and', ' AST', ',', ' which', ' suggests', ' potential', ' liver', ' damage', '.', ' Additionally', ',', ' the', ' patient', "'s", ' ultrasound', ' showed', ' a', ' fatty', ' liver', '.', '<|im_end|>', '\n', '<|im_start|>', 'assistant', '\n', 'The', ' patient', ' has', ' signs', ' of', ' liver', ' damage', ' and', ' a', ' fatty', ' liver', '.', '<|im_end|>', '<|endoftext|>']
    =========
    ['<|im_start|>', 'system', 'Ċ', 'You', 'Ġare', 'Ġa', 'Ġhelpful', 'Ġassistant', '<|im_end|>', 'Ċ', '<|im_start|>', 'user', 'Ċ', 'Ident', 'ify', 'Ġthe', 'Ġmain', 'Ġconclusion', 'Ġfrom', 'Ġthe', 'Ġprovided', 'Ġmedical', 'Ġreport', 'Ġexcerpt', '.Ċ', 'The', 'Ġpatient', "'s", 'Ġblood', 'Ġtest', 'Ġresults', 'Ġshowed', 'Ġan', 'Ġelevation', 'Ġin', 'Ġliver', 'Ġenzymes', ',', 'Ġspecifically', 'ĠALT', 'Ġand', 'ĠAST', ',', 'Ġwhich', 'Ġsuggests', 'Ġpotential', 'Ġliver', 'Ġdamage', '.', 'ĠAdditionally', ',', 'Ġthe', 'Ġpatient', "'s", 'Ġultrasound', 'Ġshowed', 'Ġa', 'Ġfatty', 'Ġliver', '.', '<|im_end|>', 'Ċ', '<|im_start|>', 'assistant', 'Ċ', 'The', 'Ġpatient', 'Ġhas', 'Ġsigns', 'Ġof', 'Ġliver', 'Ġdamage', 'Ġand', 'Ġa', 'Ġfatty', 'Ġliver', '.', '<|im_end|>', '<|endoftext|>']
    =========
    <|im_start|>system
    You are a helpful assistant<|im_end|>
    <|im_start|>user
    Identify the main conclusion from the provided medical report excerpt.
    The patient's blood test results showed an elevation in liver enzymes, specifically ALT and AST, which suggests potential liver damage. Additionally, the patient's ultrasound showed a fatty liver.<|im_end|>
    <|im_start|>assistant
    The patient has signs of liver damage and a fatty liver.<|im_end|><|endoftext|>
    =========
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    =========
    system
    You are a helpful assistant<|im_end|>
    <|im_start|>user
    Identify the main conclusion from the provided medical report excerpt.
    The patient's blood test results showed an elevation in liver enzymes, specifically ALT and AST, which suggests potential liver damage. Additionally, the patient's ultrasound showed a fatty liver.<|im_end|>
    <|im_start|>assistant
    The patient has signs of liver damage and a fatty liver.<|im_end|><|endoftext|>

    =========
    <|im_start|>system
    You are a helpful assistant<|im_end|>
    <|im_start|>user
    Identify the main conclusion from the provided medical report excerpt.
    The patient's blood test results showed an elevation in liver enzymes, specifically ALT and AST, which suggests potential liver damage. Additionally, the patient's ultrasound showed a fatty liver.<|im_end|>
    <|im_start|>assistant
    The patient has signs of liver damage and a fatty liver.<|im_end|><|endoftext|> 
    """

    batch_size = 2
    lr = 1e-5
    num_epochs = 3
    logging_steps = 1
    max_grad_norm = 1.0
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: collate_fn(batch, tokenizer))
    for batch in dataloader:
        print(f"{batch['input_ids'].shape=}")
        print(f"{batch['attention_mask'].shape=}")
        print(f"{batch['labels'].shape=}")
        break

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    model.train() # train_mode

    total_loss = 0
    total_step = 0
    for epoch in range(num_epochs):
        for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad() # 清空梯度
            outputs = model.forward(input_ids, attention_mask=attention_mask, labels=labels)

            # 注意：
            # hf_loss = outputs.loss, 原始的hf计算的loss是未将input_ids与labels移位的loss,我们手动移位后，不能使用原始loss,需要自己再算masked_cross_entropy_loss

            logits = outputs.logits
            batch, seq_len, vocab_size = logits.shape
            #from einops import rearrange
            # logits:[batch, seq_len, vocab_size]
            # rearrnged_logits:[batch*seq_len, vocab_size],普通的view就可以，没必要用rearrange吧
            #rearranged_logits = rearrange(logits, 'bsz seq_len vocab_size -> (bsz seq_len) vocab_size')
            rearranged_logits = logits.reshape(batch*seq_len, vocab_size)

            # attention_mask:[batch, seq_len]
            # rearranged_attention_mask:[batch*seq_len]
            #rearranged_attention_mask = rearrange(attention_mask, 'bsz seq_len -> (bsz seq_len)')
            rearranged_attention_mask = attention_mask.reshape(batch*seq_len)
            # labels:[batch, seq_len]
            # rearranged_labels:[batch*seq_len]
            #rearranged_labels = rearrange(labels, 'bsz seq_len -> (bsz seq_len)')
            rearranged_labels = labels.reshape(batch*seq_len)

            """
            要注意：我们的数据集是单轮问答 QA 形式，不同于预训练对所有位置都计算 loss，在做 SFT 的时候，我们只计算 Answer 部分的 loss，Question 部分我们会 mask 掉。
            当然，对于小模型的 SFT，也有做法是和预训练一样，全部位置都计算 loss，我们这里只是用最经典的方式，不关注太多其他技巧。
            
            在 Hugging Face 的实现里，training 时已经实现好了偏移一位的逻辑，不需要我们再手动实现了。我们也可以在 transformers 的源码modeling_llama里看到这一点
            所以尽管 outputs = model(input_ids, attention_mask=attention_mask, labels=labels) 的 outputs 中有 loss，我们也不能直接拿来用.
            """
            # rearranged_attention_mask:[batch*seq_len]
            # rearranged_labels:[batch*seq_len]
            # sum_loss:[batch*seq_len]
            sum_loss = F.cross_entropy(rearranged_logits, rearranged_labels, ignore_index=0, reduction='none') # 这里的ignore_index没啥用，因为后面有padding矩阵来计算loss
            # 只对padding==1的地方计算loss，padding为0的地方不计算loss
            # 按照 mask 手动计算 loss
            loss = torch.sum(sum_loss * rearranged_attention_mask) / torch.sum(rearranged_attention_mask)
            loss.backward()

            # 计算梯度范数并裁剪
            # 将所有参拼成一个向量计算2范数，即sqrt(x*(x.T)),
            # 另外这里的total_norm是在clip之前的范数，而非clip之后的，clip之后的最大不会超过max_grad_norm=1, 注意是原地clip
            total_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm, norm_type=2.0)

            optimizer.step() # 优化器每次将梯度应用于weight上
            total_loss += loss.detach().item()

            total_step += 1
            if total_step % logging_steps == 0:
                avg_loss = total_loss / total_step
                print(f"Step: {step + 1}/{len(dataloader)}, Loss: {avg_loss:.4f}, Grad Norm: {total_norm:.4f} lr:{optimizer.param_groups[0]['lr']}", flush=True)

        # 打印每个 epoch 结束的累计损失
        print(f"Epoch {epoch + 1} finished, Average Loss: {total_loss / total_step:.4f}", flush=True)

    #model.save_pretrained(output_dir)
    save_lora_model(model, "only_lora.pt")
    test_inference_stream(model, tokenizer)

def test_inference_stream(model, tokenizer,
                          test_text_list=['Describe the process of bacterial conjugation and its significance in the context of antibiotic resistance.',
                                          'Explain the role of insulin in the body and how insulin resistance affects blood sugar levels.',
                                          'Provide recommendations for lifestyle changes that can help improve the overall health of a patient with type 2 diabetes.']):
    model.eval()

    for test_text in test_text_list:
        print('=' * 80)
        last_text = ''
        for text in chatglm_inference(model, tokenizer, test_text, streaming=True):
            cur_text = text.replace(last_text, '')
            print(cur_text, end='', flush=True)
            last_text = text
        print('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--enable_wandb', type=bool, default=False)
    parser.add_argument('--is_debug', type=bool, default=True)
    parser.add_argument('--tokenizer_path', type=str, default="/home/hkx/data/work/hf_data_and_model/models/NousResearch/Llama-2-7b-hf/")
    parser.add_argument('--dataset_path', type=str, default="/home/hkx/data/work/hf_data_and_model/datas/TinyStoriesV2/")
    parser.add_argument('--output_path', type=str, default="./")

    args = parser.parse_args()
    print("parsed args:", args)

    train_lora(args)
