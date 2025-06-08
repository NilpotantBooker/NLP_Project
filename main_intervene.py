from __future__ import annotations
"""
main_intervene.py  ·  多层融合 + 语义残差 + L2 归一化

"""

import argparse, math, torch, numpy as np
import json
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM

# STOPWORDS = {"am","is","are"}
STOPWORDS = {  # 略，可自行扩充
    "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
    "yourself","yourselves","he","him","his","himself","she","her","hers",
    "herself","it","its","itself","they","them","their","theirs","themselves",
    "what","which","who","whom","this","that","these","those","am","is","are",
    "was","were","be","been","being","have","has","had","having","do","does",
    "did","doing","a","an","the","and","but","if","or","because","as","until",
    "while","of","at","by","for","with","about","against","between","into",
    "through","during","before","after","above","below","to","from","up","down",
    "in","out","on","off","over","under","again","further","then","once","here",
    "there","when","where","why","how","all","any","both","each","few","more",
    "most","other","some","such","no","nor","not","only","own","same","so",
    "than","too","very","can","will","just","should","now"
}
PUNCT = {".",",","!","?",";",":","\"","'","(",")","[","]","{","}"}
STOPWORDS |= PUNCT

TEMPLATES = {
    "PromptEOL":  'This sentence : "{sent}" means in one word: "',
    "CoT":        'After thinking step by step, this sentence : "{sent}" means in one word: "',
    "KE":         ('The essence of a sentence is often captured by its main subjects and actions, '
                   'while descriptive terms provide additional but less central details. '
                   'With this in mind, this sentence : "{sent}" means in one word: "'),
    "IntentFocus":'The primary intent or main point of the sentence "{sent}", if summarized into one word, would be: "'
}

class OneTokenCompressor:
    def __init__(self, model_name:str, prompt_style:str="PromptEOL", layer_idx:int=-1,
                 pooling:str="last", beta:float=0.7,
                 use_idf:bool=False, alpha:float=0.5, device:str|None=None):
        assert prompt_style in TEMPLATES
        assert pooling in {"last","last4_mean","last4_weighted"}
        self.style, self.LAYER = prompt_style, layer_idx
        self.pooling, self.beta, self.use_idf, self.alpha = pooling, beta, use_idf, alpha
        self.tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", output_hidden_states=True,
            output_attentions=True, low_cpu_mem_usage=True
        ).to(device or ("cuda" if torch.cuda.is_available() else "cpu")).eval()
        self.idf: Dict[str,float]|None = None

    # ---------- IDF ----------
    def build_idf(self, samples:List[str|List[str]]):
        df, docs = {}, []
        for s in samples:
            text = " ".join(s) if isinstance(s,list) else str(s)
            words=[w.lower().strip('".,!?;:()[]{}') for w in text.split()]
            words=[w for w in words if w and not all(c in PUNCT for c in w)]
            docs.append(set(words))
        N=len(docs)
        for d in docs:
            for w in d: df[w]=df.get(w,0)+1
        self.idf={w: math.log((N+1)/(c+1)) for w,c in df.items()}

    # ---------- 编码 ----------
    @torch.inference_mode()
    def encode(self,sents:List[str])->np.ndarray:
        prompts=[TEMPLATES[self.style].format(sent=s) for s in sents]
        toks=self.tok(prompts,padding=True,truncation=True,return_tensors="pt");
        toks={k:v.to(self.model.device) for k,v in toks.items()}; ids=toks["input_ids"]
        out=self.model(**toks)
        hstates = out.hidden_states          # tuple(len=L+1)
        attn    = out.attentions[self.LAYER] # (B,H,L,L)
        diag    = attn.mean(dim=1).diagonal(dim1=-2,dim2=-1)  # (B,L)
        # ---- bias (IDF / 停用词) ----
        if self.use_idf:
            mask=toks["attention_mask"]; B,L=mask.shape; bias=torch.zeros_like(diag)
            quote=self.tok.encode('"',add_special_tokens=False)[0]
            for i in range(B):
                pos=(ids[i]==quote).nonzero(as_tuple=True)[0]; s,e=(pos[0]+1,pos[1]) if len(pos)>=2 else (0,L)
                for j in range(L):
                    if not mask[i,j]: continue
                    tok=self.tok.decode([ids[i,j]]).lower().strip('".,!?;:()[]{}')
                    if j<s or j>=e or tok in STOPWORDS or tok=="":
                        bias[i,j]=-1.0
                    else:
                        bias[i,j]=self.idf.get(tok,0.0) if self.idf else 0.0
            diag = diag * bias.exp()
        # ---- 归一化权重 ----
        w = diag / (diag.sum(dim=-1,keepdim=True)+1e-8)          # (B,L)
        hidden_target = hstates[self.LAYER]                       # (B,L,D)
        g = torch.bmm(w.unsqueeze(1), hidden_target).squeeze(1)   # (B,D)
        # ---- 取 h_last (多层池化) ----
        B = ids.size(0); mask=toks["attention_mask"]; last = mask.sum(dim=-1)-1
        def last_from(layer_tensor):
            return layer_tensor[torch.arange(B,device=layer_tensor.device), last]
        if self.pooling=="last":
            h = last_from(hstates[self.LAYER])
        elif self.pooling=="last4_mean":
            layers=[hstates[self.LAYER - i] for i in range(4)]
            h = torch.stack([last_from(x) for x in layers]).mean(dim=0)
        else:  # last4_weighted  (weights 1,2,3,4)
            coeff=torch.tensor([1,2,3,4],dtype=hidden_target.dtype,device=hidden_target.device).view(4,1,1)
            layers=[last_from(hstates[self.LAYER - i]) for i in range(4)]  # list 4*(B,D)
            h = (torch.stack(layers)*coeff).sum(dim=0)/coeff.sum()
        # ---- β·h + (1-β)·g + α·g (残差) ----
        emb = self.beta*h + (1-self.beta)*g
        emb = emb + self.alpha * g if self.alpha else emb
        # ---- L2 normalize ----
        emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
        return emb.to(torch.float32).cpu().numpy()

# ---------- SentEval 接口 ----------

def build_senteval(style, model, layer, pooling, beta, idf_flag, alpha):
    enc = OneTokenCompressor(model, style, layer, pooling, beta, idf_flag, alpha)
    def prepare(params, samples):
        if idf_flag: enc.build_idf(samples)
    def batcher(params, batch):
        sentences=[" ".join(s) if isinstance(s,list) else s for s in batch]
        return enc.encode(sentences)
    return prepare, batcher

# ---------- CLI ----------
if __name__ == "__main__":
    p=argparse.ArgumentParser();
    p.add_argument("--encoder",choices=list(TEMPLATES),default="PromptEOL");
    p.add_argument("--model",default="Qwen/Qwen2-7B-Instruct");
    p.add_argument("--layer_index",type=int,default=-1);
    p.add_argument("--pooling",choices=["last","last4_mean","last4_weighted"],default="last4_mean");
    p.add_argument("--beta",type=float,default=0.7,help="权重系数 β (0~1)");
    p.add_argument("--use_idf_attention",action="store_true");
    p.add_argument("--alpha",type=float,default=0.3,help="残差注入强度 α");
    p.add_argument("--task_path",default="./SentEval/data");
    p.add_argument("--tasks",nargs="*",default=["STSBenchmark"]);
    args=p.parse_args()

    prep,batch = build_senteval(args.encoder,args.model,args.layer_index,
                                args.pooling,args.beta,args.use_idf_attention,args.alpha)

    import SentEval.senteval as se
    se_param={"task_path":args.task_path,"usepytorch":True,"batch_size":16,
              "classifier":{"nhid":0,"optim":"adam","tenacity":3,"epoch_size":2}}
    engine = se.engine.SE(se_param,batch,prep)
    results = engine.eval(args.tasks)
    print(json.dumps(results, indent=2, ensure_ascii=False))
