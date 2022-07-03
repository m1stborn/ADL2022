from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    GPT2Model,
    GPT2LMHeadModel,
    AutoModelForSequenceClassification,
    BertForPreTraining,
    BertModel,
    T5Model,
    SegformerModel,
    SegformerConfig,
)

if __name__ == '__main__':
    # Bert
    # model_name = "bert-base-cased"
    # config = AutoConfig.from_pretrained(model_name)
    # model = BertModel(config)
    # output = model(**model.dummy_inputs)
    # print(output.last_hidden_state.size())
    # print(output.pooler_output.size())
    # model = AutoModelForSequenceClassification.from_pretrained(
    #     model_name,
    #     config=config
    # )
    # print(model.main_input_name)
    # output = model(**model.dummy_inputs)
    # print(output)

    # GPT2
    # model_name = "gpt2"
    # config = AutoConfig.from_pretrained(model_name, output_attentions=True)
    # model = GPT2LMHeadModel(config)
    # output = model(**model.dummy_inputs)
    # print(output.logits.size())

    # gpt_model = GPT2Model(config)
    # output = gpt_model(**gpt_model.dummy_inputs)
    # # print(config)
    # # print(gpt_model.dummy_inputs)
    # # print(output.attentions.keys())
    # name = [type(e) for e in output.attentions]
    # print(name)
    # print(len(name))

    # T5-small
    # model_name = "t5-small"
    # config = AutoConfig.from_pretrained(model_name)
    #
    # t5_model = T5Model(config)
    # output = t5_model(**t5_model.dummy_inputs)
    # print(output.last_hidden_state.size())
    #
    # from transformers import T5Tokenizer, T5Model
    #
    # tokenizer = T5Tokenizer.from_pretrained("t5-small")
    # model = T5Model.from_pretrained("t5-small")
    #
    # input_ids = tokenizer(
    #     "Studies have been shown that owning a dog is good for you", return_tensors="pt"
    # ).input_ids  # Batch size 1
    # decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  # Batch size 1
    #
    # # forward pass
    # outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
    # print(output)

    # SegFormer
    # model_name = ""
    config = SegformerConfig()
    seg_model = SegformerModel(config)
    print(seg_model.dummy_inputs)
