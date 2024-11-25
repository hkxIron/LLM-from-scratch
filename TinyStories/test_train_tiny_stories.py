from train_tiny_stores import *
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 设置 device，能用 cuda 就用 cuda，苹果 M 系列可以用 mps
#tokenizer = AutoTokenizer.from_pretrained('/home/hkx/data/work/hf_data_and_model/models/NousResearch/Llama-2-7b-hf')

max_seq_len = 2048
hidden_size = 256
intermediate_size = (int(hidden_size * 8 / 3 / 128) + 1) * 128  # 一般为hidden_size的4倍,此处设为128的8/3倍
num_hidden_layers=4

def test_my_data_collator():
    tokenizer = AutoTokenizer.from_pretrained('/home/hkx/data/work/hf_data_and_model/models/NousResearch/Llama-2-7b-hf')
    # DataCollatorForLanguageModeling
    # 这⾥的 tokenizer 选⽤的是 Qwen1.5 的，并⾮ LLaMA 的，只是做⼀个⽰意
    my_data_collator = lambda x: batch_padding(x, tokenizer)
    data = ['南京', '南京市', '南京市⻓江']
    raw_tokens = [tokenizer(text) for text in data]
    print(f'tokenizer.pad_token_id: {tokenizer.pad_token_id}\n eos:{tokenizer.eos_token_id}')  # pad:0, eos:2


    print("raw tokens:")
    print(raw_tokens)
    """
    raw tokens:
[{'input_ids': [1, 29871, 30601, 30675], 'attention_mask': [1, 1, 1, 1]}, 
{'input_ids': [1, 29871, 30601, 30675, 30461], 'attention_mask': [1, 1, 1, 1, 1]}, 
{'input_ids': [1, 29871, 30601, 30675, 30461, 229, 190, 150, 30775], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1]}]
after collator:
{'input_ids': tensor([[    0,     0,     0,     0,     0,     1, 29871, 30601, 30675],
        [    0,     0,     0,     0,     1, 29871, 30601, 30675, 30461],
        [    1, 29871, 30601, 30675, 30461,   229,   190,   150, 30775]]), 
 'attention_mask': tensor([[0, 0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1]])}
    """

    decode_text = tokenizer.batch_decode(raw_tokens[0]['input_ids'], skip_special_tokens=False)
    print(f"decode text:{decode_text}")
    decode_text = tokenizer.batch_decode(raw_tokens[0]['input_ids'], skip_special_tokens=True)
    print(f"decode text:{decode_text}")
    decode_text = tokenizer.convert_ids_to_tokens(raw_tokens[0]['input_ids'], skip_special_tokens=True)
    print(f"decode text:{decode_text}")
    """
    decode text:['<s>', '', '南', '京'] # skip_special_tokens=False
    decode text:['', '', '南', '京'] # skip_special_tokens=True, special token <s> is removed
    decode text:['▁', '南', '京'] # skip_special_tokens=True, special token <s> is removed
    """


    print("after collator:")
    print(my_data_collator(raw_tokens))
    """
    {'input_ids':
     tensor([[    0,     0,     0,     0,     0,     1, 29871, 30601, 30675],
            [    0,     0,     0,     0,     1, 29871, 30601, 30675, 30461],
            [    1, 29871, 30601, 30675, 30461,   229,   190,   150, 30775]]), 
      'attention_mask': 
      tensor([[0, 0, 0, 0, 0, 1, 1, 1, 1],
              [0, 0, 0, 0, 1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1, 1, 1, 1, 1]])}
    """


def test_data_collator():
    tokenizer = AutoTokenizer.from_pretrained('/home/hkx/data/work/hf_data_and_model/models/NousResearch/Llama-2-7b-hf')
    # DataCollatorForLanguageModeling
    # 这⾥的 tokenizer 选⽤的是 Qwen1.5 的，并⾮ LLaMA 的，只是做⼀个⽰意
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    data = ['南京', '南京市', '南京市⻓江']
    raw_tokens = [tokenizer(text) for text in data]
    print(f'tokenizer.pad_token_id: {tokenizer.pad_token_id}\n')
    print("raw tokens:")
    print(raw_tokens)
    """
    raw_tokens:
    [{'input_ids': [1, 29871, 30601, 30675], 'attention_mask': [1, 1, 1, 1]}, 
     {'input_ids': [1, 29871, 30601, 30675, 30461], 'attention_mask': [1, 1, 1, 1, 1]}, 
     {'input_ids': [1, 29871, 30601, 30675, 30461, 229, 190, 150, 30775], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1]}
    ]


    after collator: 
    {'input_ids': 
        tensor([[    0,     0,     0,     0,     0,     1, 29871, 30601, 30675],
            [    0,     0,     0,     0,     1, 29871, 30601, 30675, 30461],
            [    1, 29871, 30601, 30675, 30461,   229,   190,   150, 30775]]), 
    'attention_mask': tensor([[0, 0, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1]]),
    'labels': tensor([[ -100,  -100,  -100,  -100,  -100,     1, 29871, 30601, 30675],
            [ -100,  -100,  -100,  -100,     1, 29871, 30601, 30675, 30461],
            [    1, 29871, 30601, 30675, 30461,   229,   190,   150, 30775]])
    }
    """

    print("after collator:")
    print(data_collator(raw_tokens))


def test_inference():
    tokenizer = AutoTokenizer.from_pretrained('/home/hkx/data/work/hf_data_and_model/models/NousResearch/Llama-2-7b-hf')
    model = AutoModelForCausalLM.from_pretrained('my_model/').to(device)
    result = inference(model, tokenizer,
                       input_text="Once upon a time, in a beautiful garden, there lived a little rabbit named Peter Rabbit.",
                       max_new_tokens=512)
    print(result)

    """
    经过14小时预训练的(20M参数个数)模型，能够生成连贯的句子，但是前后句之间仍有逻辑不一致性。

    Once upon a time, in a beautiful garden, there lived a little rabbit named Peter Rabbit. Peter loved to eat carrots and bananas. One day, Peter found a big carrot in the garden. He was so happy and wanted to take it home.
    Honey hopped around and said to Peter, "Wow, you have a carrot! I found it! Can I have it, please?" Peter smiled and said, "Yes, you can have it, but please don't eat it."
    So Peter and Rabbit went to the garden and started to eat the carrot. But it was so tasty and they were both very full. Suddenly, a big gust of wind came and blew the carrot away. Peter and Rabbit were very sad.
    Peter said, "Let's find a way to get that toy back." So Peter and Rabbit searched together and found a long stick. They found a long stick and carried it back to the garden. They were very happy and shared the carrot.
    The moral of the story is that it's important to be careful with things that can hurt you and others.
    """


if __name__ == "__main__":
    test_my_data_collator()
    if False:
        load_model_and_infer()
        test_data_collator()
        test_inference()
