from transformers import AutoTokenizer
import json
import tqdm
p = "/home/ubuntu/data/pycharm/LLaMA-Factory/data/summary/SAMSUM_test_instruction.json"
data = json.load(open(p,mode="r"))
tokenizer = AutoTokenizer.from_pretrained("/home/ubuntu/data/qwen-moe")
sum = []

for d in data:
    sum.append(len(tokenizer.encode( d['instruction'] + d['input'])))
sum = sorted(sum)
print(max(sum))
print(sum[-5:])