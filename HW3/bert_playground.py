from transformers import BertTokenizer, BertModel
import torch

if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    model = BertModel.from_pretrained("bert-base-uncased")

    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    outputs = model(**inputs)
    print(tokenizer.cls_token)
    print(tokenizer.cls_token_id)
    print(inputs)
    print(outputs[0].size(), outputs[1].size())
