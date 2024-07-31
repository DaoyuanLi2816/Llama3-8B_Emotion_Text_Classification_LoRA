import pandas as pd
from sklearn.metrics import accuracy_score

path = "/home/ubuntu/data/pycharm/LLaMA-Factory/saves/ChatGLM3-6B-Base/lora/eval_2024-02-22-09-00-36/generated_predictions.jsonl"
path = "/home/ubuntu/data/pycharm/LLaMA-Factory/saves/ChatGLM3-6B-Base/lora/eval_2024-02-22-10-18-46/generated_predictions.jsonl"
data = open(path,mode="r").readlines()
predict = []
labels = []
sum = 0
for d in  data:
    dd = eval(d)
    sum += int(dd['label'].lower() == dd['predict'].lower())
print(sum/len(data))