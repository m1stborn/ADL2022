from random import shuffle
import sys
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch
from tqdm import tqdm
from datasets import load_dataset

device = "cuda"
# gpt2, microsoft/DialoGPT-medium
model_id = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
# in domain
# prediction = load_dataset("text", data_files={"prediction": "./ckpt/concatenate.txt"})["prediction"]
# prediction = load_dataset("text", data_files={"prediction": "./ckpt/ce4e62db/concatenate.txt"})["prediction"]
prediction = load_dataset("text", data_files={"prediction": "./ckpt/3cfc44a3/concatenate.txt"})["prediction"]
# perplexity: 39.95638656616211
# sacrebleu: 8.014

# out_of_domain
# prediction = load_dataset("text", data_files={"prediction": "./ckpt/ebcef9a8/concatenate.txt"})["prediction"]
# perplexity: 41.59463119506836
# sacrebleu: 6.781

print(prediction[0])
loss = 0
steps = 0
nll = 0

for p in tqdm(prediction["text"]):
    if p:
        input_ids = tokenizer(p, return_tensors="pt").input_ids.to(device)
        #        shuffle(input_ids[0])
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            nll += outputs[0].mean().item()
    steps += 1

average_nll = nll / steps
ppl = torch.exp(torch.tensor(average_nll)).item()

print(ppl)
