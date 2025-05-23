from __future__ import annotations
import argparse, json, os
import SentEval.senteval as senteval
from prompt_encoders import build_senteval_interface

# ------------------------------ 参数 ------------------------------------ #
parser = argparse.ArgumentParser()
parser.add_argument("--encoder", choices=["PromptEOL", "CoT", "KE"],
                    default="PromptEOL", help="选择提示模板")
parser.add_argument("--model", default="Qwen/Qwen2-7B-Instruct",
                    help="HF 模型名或本地路径")
parser.add_argument("--task-path", default="./SentEval/data",
                    help="SentEval 数据集根目录 (运行 get_transfer_data.bash 后的位置)")
parser.add_argument("--tasks", nargs="*",
                    default= 'CR',
                    help="要评测的任务列表")
args = parser.parse_args()

print(f"\n>>> Running SentEval for encoder = {args.encoder}")
prepare, batcher = build_senteval_interface(args.encoder, args.model)

# --------------------------- SentEval 设置 -------------------------------- #
params_senteval = {"task_path": args.task_path,
                   "usepytorch": True,
                   "kfold": 5,
                   "batch_size": 16,
                   "classifier": {"nhid": 0, "optim": "adam", "tenacity": 3,
                                  "epoch_size": 2}}
se = senteval.engine.SE(params_senteval, batcher, prepare)

results = se.eval(args.tasks)
print(json.dumps(results, indent=2, ensure_ascii=False))

# 可选：保存结果
out_dir = "results"
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, f"{args.encoder}.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
