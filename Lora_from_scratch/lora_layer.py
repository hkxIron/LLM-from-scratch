import copy
from typing import *

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True

class LoraLinear(nn.Module):
    def __init__(
        self,
        base_layer: nn.Linear,      # 原来的线性层
        rank: int = 8,                 # lora rank
        alpha: int = 16,            # lora alpha
        dropout_p: float = 0.0,     # lora dropout
        test_mode: bool = False,    # 测试模式，用于控制 lora_B 是否为全零
    ):
        super(LoraLinear, self).__init__()
        self.base_layer = copy.deepcopy(base_layer)
        self.rank = rank
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout_p)

        # 定义 lora_A 和 lora_B 为 Parameter, 注意初始化为empty
        # 它创建一个指定大小的张量，但不会对张量的元素进行初始化, 张量中的元素的值取决于张量所在内存的状态，因此这个张量的值可能是随机的
        self.lora_A = nn.Parameter(torch.empty((rank, base_layer.in_features), dtype=base_layer.weight.dtype)) # [rank, in_feat]
        self.lora_B = nn.Parameter(torch.empty((base_layer.out_features, rank), dtype=base_layer.weight.dtype)) # [out_feat, rank]

        # 初始化 lora 矩阵
        nn.init.normal_(self.lora_A, mean=0.0, std=0.02)
        if test_mode: # 如果是测试，用正态分布初始化
            nn.init.normal_(self.lora_B, mean=0.0, std=0.02)
        else: # 如果不是测试，用全0初始化
            nn.init.zeros_(self.lora_B)

        # 冻结原来的层的参数, 原始层需要冻结参数
        for param in self.base_layer.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scaling = float(self.alpha) / float(self.rank)     # lora 缩放系数
        # down_proj = F.linear(self.dropout(x), self.lora_A)
        # up_proj = F.linear(down_proj, self.lora_B)

        # 先dropout,再使用lora
        # down_proj = X*Wa.T
        # X:[batch, seq_len, hidden]
        # Wa:[rank, in_feat=hidden],
        # down_proj: [batch, seq_len, rank]
        down_proj = self.dropout(x) @ self.lora_A.T

        # up_proj = down_proj * Wb.T
        # down_proj:[batch, seq_len, rank], Wb:[hidden, in_feat=rank]
        # up_proj:[batch, seq_len, hidden]
        up_proj = down_proj @ self.lora_B.T

        return self.base_layer(x) + up_proj * scaling


def replace_linear_with_lora(
    module: nn.Module,
    rank: int = 8,
    alpha: int = 16,
    dropout_p: float = 0.0,
    embed_requires_grad: bool = False,      # embedding 层是否训练
    norm_requires_grad: bool = False,       # norm 层是否训练
    head_requires_grad: bool = False,       # lm_head 层是否训练（Causal LM才有）
    test_mode: bool = False,                # 测试模式，用于控制 lora_B 是否为全零
):
    """
    找到 module 中所有线性层并递归替换, 原地替换

    nn.Module 的 named_children() 会返回名称和下一级的子模块。注意，这不会递归返回全部子模块，named_modules() 才是递归返回所有的子模块。
    所以我们需要递归向下查找替换。（注意：这里只考虑了语言模型，如果涉及到卷积、池化等其它类型的参数模块，
    需要在递归的时候默认把当前参数冻结，遇到 LoRA 替换才开启。）

    为什么不用 named_modules()？ 因为我们不仅仅是遍历，还需要替换参数，如果根据 named_modules() 来遍历，就会出现经典的迭代中变更元素的错误，所以我们手动向下 DFS。
    
  比如：对以下的llama model一直进行遍历，直到遇到linear层才进行lora替换
  LlamaModel(
  (embed_tokens): Embedding(128, 24)
  (layers): ModuleList(
    (0-3): 4 x LlamaDecoderLayer(
      (self_attn): LlamaSdpaAttention(
        (q_proj): Linear(in_features=24, out_features=24, bias=False)
        (k_proj): Linear(in_features=24, out_features=12, bias=False)
        (v_proj): Linear(in_features=24, out_features=12, bias=False)
        (o_proj): Linear(in_features=24, out_features=24, bias=False)
        (rotary_emb): LlamaRotaryEmbedding()
      )
      (mlp): LlamaMLP(
        (gate_proj): Linear(in_features=24, out_features=96, bias=False)
        (up_proj): Linear(in_features=24, out_features=96, bias=False)
        (down_proj): Linear(in_features=96, out_features=24, bias=False)
        (act_fn): SiLU()
      )
      (input_layernorm): LlamaRMSNorm((24,), eps=1e-06)
      (post_attention_layernorm): LlamaRMSNorm((24,), eps=1e-06)
    )
  )
  (norm): LlamaRMSNorm((24,), eps=1e-06)
  (rotary_emb): LlamaRotaryEmbedding()
    """

    for name, child in module.named_children():
        # 先处理额外的层，lm_head 也是 linear，所以先处理
        # 名称中包含embed, norm, lm_head的层
        if any(s in name for s in ['embed', 'norm', 'lm_head']):
            requires_grad = embed_requires_grad if 'embed' in name \
                            else norm_requires_grad if 'norm' in name \
                            else head_requires_grad
            # 这些层在lora微调时一般无需更新梯度
            for param in child.parameters():
                param.requires_grad = requires_grad
        # 替换所有线性层，QLoRA 做法
        elif isinstance(child, nn.Linear):
            lora_linear = LoraLinear(child, rank=rank, alpha=alpha, dropout_p=dropout_p, test_mode=test_mode)
            # 如果name=‘q_proj’, 即module.q_proj=lora_linear
            """
            替换前:
            Linear(in_features=24, out_features=24, bias=False)
             
            替换后:
            LoraLinear(
              (base_layer): Linear(in_features=24, out_features=24, bias=False)
              (dropout): Dropout(p=0.0, inplace=False)
            """
            setattr(module, name, lora_linear)
        # 递归向下替换
        else: # 如果是Module以及Module的子类
            replace_linear_with_lora(
                child, rank, alpha, dropout_p,
                embed_requires_grad, norm_requires_grad, head_requires_grad,
                test_mode=test_mode
            )

def unload_lora_then_save(module: nn.Module, adapter_file_name: str = 'adapter.pt'):
    """
    卸载 lora 参数，并将原模型恢复至加载 lora 前的样子
    1. lora参数保存至dict
    2. lora层替换成原始的mlp层，并将其设为参数可更新
    3. 将lora参数保存
    """
    lora_parameters = {}
    def search_lora_linear(module: nn.Module, name_prefix: List[str]):
        for name, child in module.named_children():
            new_prefix = name_prefix + [name]
            if isinstance(child, LoraLinear):
                # 保存 lora 参数
                lora_parameters['.'.join(new_prefix)] = {
                    "lora_A_weight": child.lora_A.data.cpu(),
                    "lora_B_weight": child.lora_B.data.cpu(),
                    "rank": child.rank,
                    "alpha": child.alpha,
                    "dropout_p": child.dropout.p,
                }
                # 将module的layer恢复
                setattr(module, name, child.base_layer)
            else:
                search_lora_linear(child, new_prefix)

    search_lora_linear(module, name_prefix=[])
    # 解冻原模型, 所有参数均可训练
    for name, param in module.named_parameters():
        param.requires_grad = True

    # 将lora的参数保存成dict
    torch.save(lora_parameters, f"{adapter_file_name}")
    return lora_parameters

def save_lora_model(module: nn.Module, adapter_file_name: str = 'adapter.pt'):
    """
    卸载 lora 参数，并将原模型恢复至加载 lora 前的样子
    1. lora参数保存至dict
    2. lora层替换成原始的mlp层，并将其设为参数可更新
    3. 将lora参数保存
    """
    lora_parameters = {}
    def search_lora_linear(module: nn.Module, name_prefix: List[str]):
        for name, child in module.named_children():
            new_prefix = name_prefix + [name]
            if isinstance(child, LoraLinear):
                # 保存 lora 参数
                lora_parameters['.'.join(new_prefix)] = {
                    "lora_A_weight": child.lora_A.data.cpu(),
                    "lora_B_weight": child.lora_B.data.cpu(),
                    "rank": child.rank,
                    "alpha": child.alpha,
                    "dropout_p": child.dropout.p,
                }
            else:
                search_lora_linear(child, new_prefix)

    search_lora_linear(module, name_prefix=[])

    # 将lora的参数保存成dict
    torch.save(lora_parameters, f"{adapter_file_name}")
    return lora_parameters

def load_and_merge_lora_from_file(module: nn.Module, adapter_file_name: str = 'adapter.pt'):
    """
    加载 lora 参数
    """
    lora_parameters = torch.load(f"{adapter_file_name}")

    for name, lora_params in lora_parameters.items():
        child = dict(module.named_modules())[name]
        if isinstance(child, nn.Linear):
            lora_linear = LoraLinear(child, lora_params['rank'], lora_params['alpha'], lora_params['dropout_p'])
            # 直接给data赋值
            lora_linear.lora_A.data = lora_params["lora_A_weight"].to(lora_linear.lora_A.device)
            lora_linear.lora_B.data = lora_params["lora_B_weight"].to(lora_linear.lora_B.device)

            # 名称示例：layers.0.self_attn.q_proj
            # 根据名称循环找到所需 module
            parts = name.split(".")
            obj = module
            # layers.0.self_attn
            # obj = layers.0.self_attn
            for part in parts[:-1]:  # 不包括最后一级
                obj = getattr(obj, part) # nn.Module重写了__getattr__函数,getattr(obj, '0')会从dict中取第0层,即obj[0]
            # 下面等价于： obj["q_proj"] = lora_linear
            setattr(obj, parts[-1], lora_linear)

    # 恢复原来的冻结方式，这里简单地除了 lora 全冻结
    for name, param in module.named_parameters():
        if any(s in name for s in ['embed', 'norm', 'lm_head']):
            param.requires_grad = False
    return lora_parameters

def get_simple_lora_param_desc(params: Dict[str, Any]):
    """
    获取 lora 参数的描述信息
    """
    lora_str = ""
    for layer_name, param in params.items():
        lora_str += f"{layer_name}: lora_A:{param['lora_A_weight'].shape} lora_B:{param['lora_B_weight'].shape} rank={param['rank']}, alpha={param['alpha']}, dropout_p={param['dropout_p']}\n"
    return lora_str


def print_trainable_parameters(model: nn.Module):
    """
    打印可训练参数，和 PeftModel 的方法类似
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_percentage = 100 * trainable_params / total_params

    print(f"trainable params: {trainable_params:,} || all params: {total_params:,} || trainable%: {trainable_percentage:.4f}")


def merge_lora(module: nn.Module):
    """
    将 lora 参数合并到原来的 base_layer 中，并将 lora 层替换回原来的 nn.Linear 层
    """

    def search_and_merge_lora_linear(module: nn.Module, prefix: List[str]):
        for name, child in module.named_children():
            # 名称示例：layers.0.self_attn.q_proj
            new_prefix = prefix + [name]
            if isinstance(child, LoraLinear):
                # 合并 lora 参数到 base_layer
                with torch.no_grad():
                    #lora_adjustment = torch.matmul(child.lora_B, child.lora_A) * (child.alpha / child.rank)
                    lora_adjustment = (child.lora_B @ child.lora_A) * (child.alpha / child.rank)
                    # 将lora权重加在base_layer上
                    child.base_layer.weight.add_(lora_adjustment)

                # 替换回原来的 base_layer
                setattr(module, name, child.base_layer)
            else:
                search_and_merge_lora_linear(child, new_prefix)

    search_and_merge_lora_linear(module, [])
    # 解冻原模型, 参数可更新
    for name, param in module.named_parameters():
        param.requires_grad = True
