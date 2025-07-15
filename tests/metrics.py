import math
import torch
import numpy as np

def make_ngram(sentence,n=1):
    res = []
    for i in range(len(sentence)-(n-1)):
        res += [" ".join(sentence[i:i+n])]
    return res

def cosine(reference, candidate, embeddings, text_splitter):
    cosi = torch.nn.CosineSimilarity(dim=0)
    
    res_splitted = text_splitter.split_text(candidate)
    embedding_res = embeddings.embed_documents(res_splitted)
    mean_res = torch.tensor(np.mean(np.array(embedding_res), axis=0))

    answ_splitted = text_splitter.split_text(reference)
    embedding_answ = embeddings.embed_documents(answ_splitted)
    mean_answ = torch.tensor(np.mean(np.array(embedding_answ), axis=0))

    return cosi(mean_res,mean_answ).item()

def bleu(reference, candidate):
    ref_tokens = reference.lower().split()
    cand_tokens = candidate.lower().split()
    scores = []

    for n in range(1,5):
        rscore = 0
        cscore = 0

        ref_ng = make_ngram(ref_tokens,n)
        cand_ng = make_ngram(cand_tokens,n)

        for ng in list(set(cand_ng)):
            rscore += min(ref_ng.count(ng),cand_ng.count(ng))
            cscore += cand_ng.count(ng)

        if cscore == 0: scores.append(0)
        else: scores.append(rscore/cscore)

    bp = min(1, math.exp(1 - len(ref_tokens)/len(cand_tokens)))

    return bp * math.exp(sum([math.log(s) for s in scores if s>0])/4)

def rouge(reference, candidate, n=1):
    ref_ng = make_ngram(reference.lower().split(),n)
    cand_ng = make_ngram(candidate.lower().split(),n)

    matches = [w for w in cand_ng if w in ref_ng]

    recall = len(matches)/len(ref_ng)
    precision = len(matches)/len(cand_ng)
    f1 = 2 * ((precision*recall)/(precision+recall))

    return f1