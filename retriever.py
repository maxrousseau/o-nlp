from rank_bm25 import BM25Plus
from nltk.corpus import stopwords

import logging

# retriever class #############################################################


# queries = [i["question"] for i in dataset]


class Retriever:

    # FORMAT = "[%(levelname)s] :: %(asctime)s @ %(name)s :: %(message)s"
    # logging.basicConfig(format=FORMAT)
    # logger = logging.getLogger("bert")
    # logger.setLevel(logging.DEBUG)  # change level if debug or not

    def __init__(self, corpus, corpus_keys, queries, labels=None, rm_stopwords=True):
        self.corpus = corpus
        self.corpus_keys = corpus_keys
        self.queries = queries  # @NOTE :: queries contain question only because
        # that woulld be what is used in real life
        self.labels = labels

        # @TODO :: add options to change
        # @TODO :: add stemming/lemmatization and stopword removal
        self.lemma = False
        self.rm_stopwords = rm_stopwords
        self.lowercase = True
        self.bm25 = None

        self.stopwords = set(stopwords.words("english"))

    def __preprocessCorpus(self):
        for idx in range(len(self.corpus)):
            c = self.corpus[idx]
            c = c.lower()
            c = c.split(" ")

            if self.rm_stopwords:
                c = [w for w in c if not w in self.stopwords]

            self.corpus[idx] = c

    def __preprocessQueries(self):
        for idx in range(len(self.queries)):
            q = self.queries[idx]
            q = q.lower()
            q = q.split(" ")

            if self.rm_stopwords:
                q = [w for w in q if not w in self.stopwords]

            self.queries[idx] = q

    def __initializeModel(self):
        self.bm25 = BM25Plus(self.corpus)

    def rankTopK(self, query, k):

        doc_ranks = self.bm25.get_scores(query)
        sorted_docs = sorted(
            ((value, index) for index, value in enumerate(doc_ranks)), reverse=True
        )
        top_k = sorted_docs[:k]

        ranked_keys = []  # k length list of article_ids
        for value, index in top_k:
            ranked_keys.append(self.corpus_keys[index])

        # @BUG :: check if this actually working correctly

        return ranked_keys

    def eval(self):

        # preprocess queries
        self.__preprocessQueries()
        self.__preprocessCorpus()

        # initialize model
        self.__initializeModel()

        # evaluate on queries and labels
        total = len(self.queries)
        correct_doc = 0
        k = 50

        for i in range(len(self.queries)):
            top_k = self.rankTopK(self.queries[i], k)
            top_k_ids = [i["article_id"] for i in top_k]

            if self.labels[i] in top_k_ids:
                correct_doc += 1

        score = 100 * (correct_doc / total)
        print("top-{} accuracy for retrival: {}%".format(str(k), score))


def testFunc():
    import load_data

    corpus_path = "C:/Users/roum5/source/data/angle_corpus/parsed_papers_6063.json"
    dataset_path = "C:/Users/roum5/source/data/oqa_v0.1_prelim/oqa_v0.1_retrieval.json"

    corpus, corpus_key = load_data.loadCorpus(corpus_path)
    dataset = load_data.loadOQAforRetrieval(dataset_path)

    queries = [i["question"] for i in dataset]
    labels = [i["context_key"] for i in dataset]

    rtrvr = Retriever(corpus, corpus_key, queries, labels=labels, rm_stopwords=True)
    # @NOTE :: interestingly stopword removal hurts performance at k=20 and at K=50
    rtrvr.eval()
