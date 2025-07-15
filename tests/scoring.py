from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import TokenTextSplitter, RecursiveCharacterTextSplitter 
from scanner import CustomTextSplitter

from main import answer_query
from metrics import cosine, bleu, rouge

def evaluate(model_name = "./models/Qwen2.5-7B-Instruct-merged",
             model = None,
             embeddings = None,
             text_splitter = None,
             questions = [],
             answers = [], 
             method = "all",
             make_db = False):
    if method not in ["cosine", "bleu", "rouge", "all"]: raise ValueError(f"Неизвестный метод оценки: {method}")

    db_remade = False

    if model is None:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto", local_files_only=True, load_in_4bit=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    scores = {}

    if method in ['cosine', 'all']:
        if embeddings is None: embeddings = HuggingFaceEmbeddings(model_name="./models/FRIDA")
        if text_splitter is None: text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=256)
    elif make_db:
        if text_splitter is None: text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=256)

    for question_number in range(len(questions)):
        if make_db and not(db_remade):
            result = answer_query(questions[question_number], model=model, tokenizer=tokenizer, text_splitter=text_splitter, new_db=True)
            db_remade = True
        else:
            result = answer_query(questions[question_number], model=model, tokenizer=tokenizer)

        if method in ['cosine', 'all']: 
            scores['cosine'] = scores.get("cosine", 0) + cosine(reference=answers[question_number], candidate=result, 
                                                                embeddings=embeddings, text_splitter=text_splitter)
        
        if method in ['bleu', 'all']: scores['bleu'] = scores.get("bleu", 0) + bleu(reference=answers[question_number], candidate=result)

        if method in ['rouge', 'all']: scores['rouge'] = scores.get("rouge", 0) + rouge(reference=answers[question_number], candidate=result)
            
    return {s: scores[s]/len(questions) for s in scores}

if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained("./models/Qwen2.5-7B-Instruct-merged", torch_dtype="auto", device_map="auto", local_files_only=True, load_in_4bit=True)
    embeddings = HuggingFaceEmbeddings(model_name="./models/FRIDA")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=0)

    print(evaluate(
        questions=[
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
        ],
        model=model,
        embeddings=embeddings,
        text_splitter=text_splitter,
        make_db=True
    ))