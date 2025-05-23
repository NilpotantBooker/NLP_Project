
from __future__ import annotations
import torch, numpy as np
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM


# ---- 模板字典 ----------------------------------------------------------- #
TEMPLATES = {
    "PromptEOL": 'This sentence : "{sent}" means in one word: "',
    "CoT":       'After thinking step by step , this sentence : "{sent}" means in one word: "',
    "KE":        'The essence of a sentence is often captured by its main subjects and actions, '
                 'while descriptive terms provide additional but less central details. '
                 'With this in mind , this sentence : "{sent}" means in one word: "'
}


class OneTokenCompressor:
    """
    句子语义压缩器：用 decoder-only LLM 把整句语义 squeeze 到生成起始位的 hidden_state
    """
    def __init__(self,
                 model_name: str = "Qwen/Qwen2-7B-Instruct",
                 prompt_style: str = "PromptEOL",
                 device: str = None):
        assert prompt_style in TEMPLATES, f"prompt_style must be one of {list(TEMPLATES)}"
        self.prompt_style = prompt_style
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            output_hidden_states=True,
            torch_dtype="auto",
            device_map={"": "cpu"},
            low_cpu_mem_usage=True
        )
        if device:
            self.model.to(device)
        self.model.eval()

    @torch.inference_mode()
    def encode(self, sentences: List[str]) -> np.ndarray:
        """
        ① 把每个句子填到模板
        ② 前向传播得到隐藏层
        ③ 取 **最末 token**（即将开始生成单词的位置）的指定层 hidden-state
        ④ 返回 (batch, hidden_dim) numpy 向量
        """
        prompts = [TEMPLATES[self.prompt_style].format(sent=s) for s in sentences]
        tokenized = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        tokenized = {k: v.to(self.model.device) for k, v in tokenized.items()}

        outputs = self.model(**tokenized)
        # hidden_states: tuple(n_layers, batch, seq, dim)
        hidden = outputs.hidden_states[-1]            # 最后一层，更抽象
        attention_mask = tokenized["attention_mask"]

        # 找到每个样本的最后一个非 PAD 位置
        last_idx = attention_mask.sum(dim=-1) - 1     # (batch,)
        batch_idx = torch.arange(hidden.size(0), device=hidden.device)
        vectors = hidden[batch_idx, last_idx]         # (batch, dim)
        return vectors.float().cpu().numpy()


# ---------- SentEval 接口封装 ------------------------------------------- #
def build_senteval_interface(prompt_style: str, model_name: str):
    """
    返回 prepare 与 batcher；二者将被 SentEval 调用
    """
    encoder = OneTokenCompressor(model_name=model_name, prompt_style=prompt_style)

    def prepare(params, samples):
        # 可选：在这里提前做词表统计等——本例直接返回即可
        return

    def batcher(params, batch):
        # batch 是 List[List[str]]；需转换成 List[str]
        sentences = [" ".join(s) if isinstance(s, list) else s for s in batch]
        return encoder.encode(sentences)

    return prepare, batcher
