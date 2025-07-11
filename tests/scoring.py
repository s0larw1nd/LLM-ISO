from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import TokenTextSplitter, RecursiveCharacterTextSplitter 
import torch
import numpy as np

from main import answer_query
from metrics import bleu, rouge

def evaluate(model_name="./models/Qwen2.5-7B-Instruct-merged", questions=[], answers=[], method="cosine"):
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto", local_files_only=True, load_in_4bit=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    score = 0

    if method == 'cosine':
        embedding = HuggingFaceEmbeddings(model_name="./models/FRIDA")
        cosi = torch.nn.CosineSimilarity(dim=0)
        text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=256)

    for question_number in range(len(questions)):
        result = answer_query(questions[question_number], model=model, tokenizer=tokenizer)

        match method:
            case "cosine":
                res_splitted = text_splitter.split_text(result)
                embedding_res = embedding.embed_documents(res_splitted)
                mean_res = torch.tensor(np.mean(np.array(embedding_res), axis=0))

                answ_splitted = text_splitter.split_text(answers[question_number])
                embedding_answ = embedding.embed_documents(answ_splitted)
                mean_answ = torch.tensor(np.mean(np.array(embedding_answ), axis=0))

                score += cosi(mean_res,mean_answ).item()

            case "bleu":
                score += bleu(reference=answers[question_number], candidate=result)

            case "rouge":
                score += rouge(reference=answers[question_number], candidate=result)

            case _:
                raise ValueError(f"Неизвестный метод оценки: {method}")
    
    return score/len(questions)
            
print(evaluate(questions=[
    "О чём говорится в пункте 7.1.6?",
    "Должна ли организация разрабатывать собственное производственное оборудование?"
],
answers=[
    """
Пункт 7.1.6 говорит о знаниях организации – знаниях (информации, которая используется и которой обмениваются для достижения целей организации), 
специфичных для организации и полученных в основном из опыта. Пункт описывает необходимость их определения, поддержки и обеспечения доступа в необходимом объёме. 
Организация должна определять способ получения дополнительных знаний и их необходимых обновлений исходя из изменяющихся нужд и тенденций. Основой знаний организации могут быть: 
внутренние источники (интеллектуальная собственность; знания, полученные из опыта; выводы из проектов) и внешние источники (стандарты, научное сообщество, конференции, семинары). 
Подпункты 7.1.6.1.1 и 7.1.6.1.2 говорят об обязанностях организации по отношению к знаниям.
    """,
    """
Пункт 8.5.1.4.1 говорит о том, что, при возможности, организация должна разрабатывать собственное производственное оборудование.
    """
]))