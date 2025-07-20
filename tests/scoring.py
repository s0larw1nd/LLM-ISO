from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import TokenTextSplitter, RecursiveCharacterTextSplitter 
from scanner import CustomTextSplitter
import os

from main import answer_query
from metrics import cosine, bleu, rouge

import config

def evaluate(model_name = config.DEFAULT_MODEL,
             model = None,
             embeddings = None,
             text_splitter = None,
             questions = [],
             answers = [], 
             method = "all",
             make_db = False,
             view_logs = False):
    if method not in ["cosine", "bleu", "rouge", "all"]: raise ValueError(f"Неизвестный метод оценки: {method}")

    db_remade = False

    if model is None:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto", local_files_only=True, load_in_4bit=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    scores = {}

    if method in ['cosine', 'all']:
        if embeddings is None: embeddings = HuggingFaceEmbeddings(model_name=config.DEFAULT_EMBEDDINGS)
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
            csn = cosine(reference=answers[question_number], candidate=result, embeddings=embeddings, text_splitter=text_splitter)
            scores['cosine'] = scores.get("cosine", 0) + csn
        
        if method in ['bleu', 'all']: 
            bl = bleu(reference=answers[question_number], candidate=result)
            scores['bleu'] = scores.get("bleu", 0) + bl

        if method in ['rouge', 'all']: 
            rg = rouge(reference=answers[question_number], candidate=result)
            scores['rouge'] = scores.get("rouge", 0) + rg
        
        if view_logs:
            print(questions[question_number])
            print(result)
            print()

            if csn: print(f"cosine: {csn}")
            if bl: print(f"bleu: {bl}")
            if rg: print(f"rouge: {rg}")

            print("----------------------------------------------------------------------------------")

    return {s: scores[s]/len(questions) for s in scores}

if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained("./models/Qwen2.5-7B-Instruct-merged", torch_dtype="auto", device_map="auto", local_files_only=True, load_in_4bit=True)
    embeddings = HuggingFaceEmbeddings(model_name="./models/Qwen3-Embedding-0.6B")
    text_splitter = CustomTextSplitter(chunk_size=300)

    print(evaluate(
        questions=[
        "Какие факторы организация должна учитывать при разработке стратегического направления и как они отражаются в бизнес-планировании?",
        "Как организация должна проводить проверку персонала?",
        "О чём говорится в пункте 7.1.6?",
        "Должна ли организация разрабатывать собственное производственное оборудование?",
        "Где говорится о том, что должен включать в себя процесс управления документированной информацией?",
        "Что такое FAI?"
        ],
        answers=[
        """
    Организация должна учитывать цели экономической деятельности, рыночную стратегию, стратегию развития продукции и услуг (включая план разработки новой продукции и услуг и(или) процессов, 
    нововведений и поэтапного вывода с рынка неактуальной продукции и услуг), результаты анализа со стороны руководства, планирование ресурсов, риски и новые возможности организации, непрерывность деятельности, 
    потребности и ожидания потребителей, вклад заинтересованных сторон (например, внешних поставщиков), влияние изменений технологических процессов, законодательных и нормативных правовых требований, технические 
    возможности организации с учётом прогнозных ожиданий, слияние, поглощение, привлечение внешних ресурсов и передача прав, в зависимости от ситуации. Также при бизнес-планировании следует учитывать: 
    изменение внешних тенденций и потребностей заинтересованных сторон (например, экономической политики, защиты окружающей среды, социального и культурного обеспечения населения, требований информационной 
    безопасности в отношении продукции и услуг), финансовый календарь организации, необходимость своевременного доведения до заинтересованных сторон результатов бизнес-планирования, необходимость выполнения 
    действий, определённых по результатам анализа бизнес-плана.
        """,
        """
    Организация должна проводить проверку персонала по процессам на основании установленных критериев необходимости проведения подготовки, установленных категорий персонала, нуждающихся в подготовке по процессам, 
    планов и программ подготовки, понимания слушателями содержания курсов подготовки.
        """,
        """
    Пункт 7.1.6 говорит о знаниях организации – знаниях (информации, которая используется и которой обмениваются для достижения целей организации), специфичных для организации и полученных в основном из опыта.
    Пункт описывает необходимость их определения, поддержки и обеспечения доступа в необходимом объёме. Организация должна определять способ получения дополнительных знаний и их необходимых обновлений исходя из изменяющихся нужд и тенденций. 
    Основой знаний организации могут быть: внутренние источники (интеллектуальная собственность; знания, полученные из опыта; выводы из проектов) и внешние источники (стандарты, научное сообщество, конференции, семинары). 
    Подпункты 7.1.6.1.1 и 7.1.6.1.2 говорят об обязанностях организации по отношению к знаниям.
        """,
        """
    Пункт 8.5.1.4.1 говорит о том, что, при возможности, организация должна разрабатывать собственное производственное оборудование.
        """,
        """
    Об управлении документированной информацией говорится в пункте 7.5.3.3
        """,
        """
    FAI – first article inspection (контроль первого изделия) – комплекс действий по контролю и проверке для утверждения производственного процесса.
    Организация должна разработать, внедрить и поддерживать процесс менеджмента контроля первого изделия (FAI), который включает в себя: планирование в соответствии с установленными критериями для идентификации продукции, 
    подлежащей FAI; подготовку FAI; выполнение мероприятий по инспекционному контролю и верификации, в том числе анализ производственных процессов, уделяя особое внимание критическим и специальным процессам; критерии для 
    выпуска серийной продукции, условного выпуска, браковки; контроль выполнения корректирующих действий.
    Процесс FAI должен применяться в отношении внутренних продуктов, репрезентативного изделия из первой серийной партии новой продукции или существенного изменения соответствующего продукта после верификации производственного 
    процесса или изменения, которое делает результаты предыдущей инспекции первого изделия недействительными.
        """
        ],
        model=model,
        embeddings=embeddings,
        text_splitter=text_splitter,
        make_db=True
    ))