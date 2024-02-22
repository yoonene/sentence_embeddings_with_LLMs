import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LlamaTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import time

# Import our models. The package will take care of downloading the models automatically
model_path = "beomi/open-llama-2-ko-7b"
# model_path = "42dot/42dot_LLM-PLM-1.3B"
lora_weight = ""
# lora_weight = "42dot_LLM-PLM-1.3B-lora/checkpoint-300"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
)
if "llama" in model_path:
    tokenizer = LlamaTokenizer.from_pretrained(model_path, use_fast=True)
    tokenizer.bos_token_id = 1
    tokenizer.eos_token = "</s>"
else:
    tokenizer = AutoTokenizer.from_pretrained(model_path)

tokenizer.pad_token_id = 0
tokenizer.padding_side = "left"

if lora_weight:
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, lora_weight, torch_dtype=torch.float16)

texts = [
    "스케이트보드를 타는 아이가 있다.",
    "아이가 스케이트보드를 타고 있다.",
    "아이가 집 안에 있다.",
]

# template = 'This_sentence_:_"A_jockey_riding_a_horse."_means_in_one_word:"Equestrian".This_sentence_:_"*sent_0*"_means_in_one_word:"'
# template = 'This_sentence_:_"기수가_말을_타고_있다."_means_in_one_word:"승마".This_sentence_:_"*sent_0*"_means_in_one_word:"'
template = '이 문장_:_"기수가_말을_타고_있다."_은_한_단어로:"승마".이_문장_:_"*sent_0*"_은_한_단어로:"'
inputs = tokenizer(
    [template.replace("*sent_0*", i).replace("_", " ") for i in texts],
    padding=True,
    return_tensors="pt",
)
s_time = time.time()
with torch.no_grad():
    embeddings = model(
        **inputs, output_hidden_states=True, return_dict=True
    ).hidden_states[-1][:, -1, :]
print(f"embedding time: {time.time() - s_time} sec")
print(len(embeddings))
cos_sim = cosine_similarity(embeddings)

print(cos_sim)
