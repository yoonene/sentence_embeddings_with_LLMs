import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity

from peft import PeftModel

# Import our models. The package will take care of downloading the models automatically
model_path = "42dot/42dot_LLM-PLM-1.3B"
lora_weight = "42dot_LLM-PLM-1.3B-lora/checkpoint-300"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer.pad_token_id = 0
tokenizer.padding_side = "left"
peft_model = PeftModel.from_pretrained(model, lora_weight, torch_dtype=torch.float16)

texts = [
    "스케이트보드를 타는 아이가 있다.",
    "아이가 스케이트보드를 타고 있다.",
    "아이가 집 안에 있다.",
]
template = 'This_sentence_:_"*sent_0*"_means_in_one_word:"'
# template = 'This_sentence_:_"A_jockey_riding_a_horse."_means_in_one_word:"Equestrian".This_sentence_:_"*sent_0*"_means_in_one_word:"'
inputs = tokenizer(
    [template.replace("*sent_0*", i).replace("_", " ") for i in texts],
    padding=True,
    return_tensors="pt",
)
with torch.no_grad():
    embeddings = peft_model(
        **inputs, output_hidden_states=True, return_dict=True
    ).hidden_states[-1][:, -1, :]
# inputs = tokenizer(
#     [template.replace("*sent_0*", i).replace("_", " ") for i in texts],
#     padding=True,
#     return_tensors="pt",
# )
# with torch.no_grad():
#     embeddings = model(
#         **inputs, output_hidden_states=True, return_dict=True
#     ).hidden_states[-1][:, -1, :]

print(len(embeddings))
cos_sim = cosine_similarity(embeddings)

print(cos_sim)
