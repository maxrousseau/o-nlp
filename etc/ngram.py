import os

# import load_data

from nltk.tokenize import sent_tokenize
from nltk.util import ngrams

from tqdm.auto import tqdm

fpath = "c:/Users/roum5/source/data/oqa_v0.1_prelim/oqa_v0.1_retrieval.json"


def getAnswerSentence(context, answer):
    slist = sent_tokenize(context)
    for s in slist:
        if answer in s:
            return s
        else:
            None


def findNgrams(dataset, n):
    contexts = [x.get("context").lower() for x in dataset]
    answers = [x.get("answer").lower() for x in dataset]
    sentences = []
    for i in range(len(contexts)):
        try:
            assert answers[i] in contexts[i]
            s = getAnswerSentence(contexts[i], answers[i])
            if s != None:
                sentences.append(s)
        except:
            print("ERROR at index {}".format(i))
            print(answers[i])
            print(contexts[i])
            continue

    # @NOTE ::  should punctuation be removed?
    print(len(sentences))
    print(type(sentences))
    tokenized_sentences = " ".join(sentences).split()
    return ngrams(tokenized_sentences, n)


# Usage:
# dataset = loadOQAforRetrieval(fpath)
# ng3 = findNgrams(dataset, 3)
# ng3_freq = collections.Counter(ng3)
# ng3_freq3 = [ (k, v) for k, v in dict(ng3_freq).items() if v > 2] # get items n-grams that occur 3 times or more
# @HERE -- next step search for the number of occurence of a select few of these in the corpus

# corpus_path = "C:/Users/roum5/source/data/angle_corpus/parsed_papers_6063.json"
def occurs(corpus, ngram):
    """count how many times an n-gram occurs in the reference corpus"""
    occ = 0
    ngram = ngram.lower()
    for p in tqdm(corpus):
        p = p.lower()
        occ += p.count(ngram)

    print("ngram \{}\ occurs {} times in corpus".format(ngram, occ))
