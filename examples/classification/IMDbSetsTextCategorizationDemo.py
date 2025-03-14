import argparse
import logging
import numpy as np
import keras
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import CountVectorizer
from tmu.models.classification.vanilla_classifier import TMClassifier
from scipy.sparse import csr_matrix

from tmu.tools import BenchmarkTimer

_LOGGER = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)

def metrics(args):
    return dict(
        accuracy=[],
        absorbed=[],
        unallocated=[],
        train_time=[],
        test_time=[],
        args=vars(args)
    )


def main(args):
    experiment_results = metrics(args)

    _LOGGER.info("Preparing dataset")
    train, test = keras.datasets.imdb.load_data(num_words=args.imdb_num_words, index_from=args.imdb_index_from)
    train_x, train_y = train
    test_x, test_y = test

    word_to_id = keras.datasets.imdb.get_word_index()
    word_to_id = {k: (v + args.imdb_index_from) for k, v in word_to_id.items()}
    word_to_id["<PAD>"] = 0
    word_to_id["<START>"] = 1
    word_to_id["<UNK>"] = 2
    _LOGGER.info("Preparing dataset.... Done!")

    _LOGGER.info("Producing bit representation...")

    id_to_word = {value: key for key, value in word_to_id.items()}

    training_documents = []
    for i in range(train_y.shape[0]):
        terms = []
        for word_id in train_x[i]:
            terms.append(id_to_word[word_id].lower())

        training_documents.append(terms)

    testing_documents = []
    for i in range(test_y.shape[0]):
        terms = []
        for word_id in test_x[i]:
            terms.append(id_to_word[word_id].lower())

        testing_documents.append(terms)

    vectorizer_X = CountVectorizer(
        tokenizer=lambda s: s,
        token_pattern=None,
        ngram_range=(1, args.max_ngram),
        max_features=100000,
        lowercase=False,
        binary=True
    )

    X_train = vectorizer_X.fit_transform(training_documents).astype(np.uint32)
    Y_train = train_y.astype(np.uint32)

    X_test = vectorizer_X.transform(testing_documents).astype(np.uint32)
    Y_test = test_y.astype(np.uint32)
    _LOGGER.info("Producing bit representation... Done!")

    _LOGGER.info("Selecting Features....")

    #SKB = SelectKBest(chi2, k=args.features)
    #SKB.fit(X_train, Y_train)

    selected_features = np.arange(args.features)
    #selected_features = SKB.get_support(indices=True)
    #X_train = SKB.transform(X_train).astype(np.uint32)
    #X_test = SKB.transform(X_test).astype(np.uint32)

    documents = [["movie", "all"], ["very", "good"], ["love", "the", "book"]]
    print(documents)
    concepts = vectorizer_X.transform(documents)
    print(concepts)
    
    _LOGGER.info("Selecting Features.... Done!")

    tm = TMClassifier(
        args.num_clauses,
        args.T,
        args.s,
        platform=args.platform,
        weighted_clauses=args.weighted_clauses,
        clause_drop_p=args.clause_drop_p,
        sets=concepts#csr_matrix([[1,8],[0,1],[15,128]])
    )

    for e in range(args.epochs):

        benchmark1 = BenchmarkTimer(logger=_LOGGER, text="Training Time")
        with benchmark1:
            tm.fit(X_train, Y_train)
        experiment_results["train_time"].append(benchmark1.elapsed())

        benchmark2 = BenchmarkTimer(logger=_LOGGER, text="Testing Time")
        with benchmark2:
            result = 100 * (tm.predict(X_test) == Y_test).mean()
        experiment_results["test_time"].append(benchmark2.elapsed())
        experiment_results["accuracy"].append(int(result))

        _LOGGER.info("#%d Accuracy: %.2f%% Training: %.2fs Testing: %.2fs" % (
            e + 1,
            result,
            benchmark1.elapsed(),
            benchmark2.elapsed()
        ))

    return experiment_results

def default_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_clauses", default=10000, type=int)
    parser.add_argument("--T", default=8000, type=int)
    parser.add_argument("--s", default=2.0, type=float)
    parser.add_argument("--platform", default="CPU_sets", type=str)
    parser.add_argument("--weighted_clauses", default=True, type=bool)
    parser.add_argument("--epochs", default=60, type=int)
    parser.add_argument("--clause_drop_p", default=0.75, type=float)
    parser.add_argument("--max-ngram", default=2, type=int)
    parser.add_argument("--features", default=5000, type=int)
    parser.add_argument("--imdb-num-words", default=5000, type=int)
    parser.add_argument("--imdb-index-from", default=2, type=int)
    args = parser.parse_args()
    for key, value in kwargs.items():
        if key in args.__dict__:
            setattr(args, key, value)
    return args

if __name__ == "__main__":

    results = main(default_args())
    _LOGGER.info(results)
