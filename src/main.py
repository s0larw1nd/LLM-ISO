from transformers import AutoModelForCausalLM,AutoTokenizer
import torch

model_name = "./models/Qwen2.5-7B-Instruct-merged"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    # local_files_only=True если есть локально
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "" # Вопрос
messages = [
    {"role": "system", "content": """Используя только данный контекст, дай максимально подробный ответ на вопрос о содержании документа. 
     Крайне важно: ни в коем случае не добавляй ничего не из контекста. По возможности, пиши источники информации (номера и названия пунктов документа).
     Если в тексте ответа есть ссылка на какой-либо документ, по возможности приведи нужный текст из этого документа.
     Пиши всегда на русском языке. 
КОНТЕКСТ:

КОНЕЦ КОНТЕКСТА """},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

with torch.no_grad():
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=8192
    )

generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
print(model_inputs.input_ids.shape[1])