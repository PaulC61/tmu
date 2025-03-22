import argparse
from scipy.sparse import csr_matrix

from tmu.models.classification.vanilla_classifier import TMClassifier
import numpy as np

overlap = 4

def main(args):
    concepts = np.empty((3, args.number_of_features), dtype=np.uint32)
    concepts[0,:args.number_of_features//2+overlap//2] = 1
    concepts[0,args.number_of_features//2+overlap//2:] = 0
    print(concepts[0])

    concepts[1,:args.number_of_features//2-overlap//2] = 0
    concepts[1,args.number_of_features//2-overlap//2:] = 1
    print(concepts[1])

    concepts[2] = np.maximum(concepts[0] - np.minimum(concepts[0], concepts[1]), concepts[1] - np.minimum(concepts[0], concepts[1]))
    print(concepts[2])
    
    class_0 = np.intersect1d(concepts[0].nonzero()[0], concepts[2].nonzero()[0])
    class_1 = np.intersect1d(concepts[1].nonzero()[0], concepts[2].nonzero()[0])

    print(class_0)
    print(class_1)

    # concepts[3,:args.number_of_features//2-1] = 0
    # concepts[3,args.number_of_features//2-1:] = 1
    # print(concepts[1])

    # concepts[2,:] = 1
    # concepts[2,args.number_of_features//2] = 0
    # print(concepts[2])

    # concepts[3,:] = 1
    # concepts[3,args.number_of_features//2-1] = 0
    # print(concepts[3])

    X_train = np.zeros((args.number_of_examples, args.number_of_features), dtype=np.uint32)
    Y_train = np.empty(args.number_of_examples, dtype=np.uint32)
    for i in range(args.number_of_examples):
        Y_train[i] = np.random.randint(2)
        if Y_train[i] == 1:
            X_train[i,np.random.choice(class_1)] = 1
            #print(X_train[i,:])
        else:
            X_train[i,np.random.choice(class_0)] = 1
            #print(X_train[i,:])

    Y_train = np.where(np.random.rand(args.number_of_examples) <= args.noise, 1 - Y_train, Y_train)  # Adds noise

    X_test = np.zeros((args.number_of_examples, args.number_of_features), dtype=np.uint32)
    Y_test = np.empty(args.number_of_examples, dtype=np.uint32)
    for i in range(args.number_of_examples):
        Y_test[i] = np.random.randint(2)
        #X_test[i,np.random.choice(concepts[Y_test[i],:].nonzero()[0])] = 1
        if Y_test[i] == 1:
            X_test[i,np.random.choice(class_1)] = 1
        else:
            X_test[i,np.random.choice(class_0)] = 1

    tm = TMClassifier(args.number_of_clauses, args.T, args.s, weighted_clauses=True, platform=args.platform, concept_sets=csr_matrix(concepts))

    for i in range(args.epochs):
        tm.fit(X_train, Y_train)
        accuracy = 100 * (tm.predict(X_test) == Y_test).mean()
        print("Accuracy:", accuracy)

    np.set_printoptions(threshold=np.inf, linewidth=200, precision=2, suppress=True)

    print("\nClass 0 Positive Clauses:\n")
    precision = tm.clause_precision(0, 0, X_test, Y_test)
    recall = tm.clause_recall(0, 0, X_test, Y_test)

    for j in range(args.number_of_clauses // 2):
        print("Clause #%d W:%d P:%.2f R:%.2f " % (j, tm.get_weight(0, 0, j), precision[j], recall[j]), end=' ')
        l = []
        for k in range(4):
            if tm.get_ta_action(j, k, the_class=0, polarity=0):
                if k < args.number_of_features:
                    l.append(" x%d(%d)" % (k, tm.get_ta_state(j, k, the_class=0, polarity=0)))
                else:
                    l.append("¬x%d(%d)" % (k - args.number_of_features, tm.get_ta_state(j, k, the_class=0, polarity=0)))
        print(" ∧ ".join(l))

    print("\nClass 0 Negative Clauses:\n")

    precision = tm.clause_precision(0, 1, X_test, Y_test)
    recall = tm.clause_recall(0, 1, X_test, Y_test)

    for j in range(args.number_of_clauses // 2):
        print("Clause #%d W:%d P:%.2f R:%.2f " % (j, tm.get_weight(0, 1, j), precision[j], recall[j]), end=' ')
        l = []
        for k in range(4):
            if tm.get_ta_action(j, k, the_class=0, polarity=1):
                if k < args.number_of_features:
                    l.append(" x%d(%d)" % (k, tm.get_ta_state(j, k, the_class=0, polarity=1)))
                else:
                    l.append("¬x%d(%d)" % (k - args.number_of_features, tm.get_ta_state(j, k, the_class=0, polarity=1)))
        print(" ∧ ".join(l))

    print("\nClass 1 Positive Clauses:\n")

    precision = tm.clause_precision(1, 0, X_test, Y_test)
    recall = tm.clause_recall(1, 0, X_test, Y_test)
 
    for j in range(args.number_of_clauses // 2):
        print("Clause #%d W:%d P:%.2f R:%.2f " % (j, tm.get_weight(1, 0, j), precision[j], recall[j]), end=' ')
        l = []
        for k in range(4):
            if tm.get_ta_action(j, k, the_class=1, polarity=0):
                if k < args.number_of_features:
                    l.append(" x%d(%d)" % (k, tm.get_ta_state(j, k, the_class=1, polarity=0)))
                else:
                    l.append("¬x%d(%d)" % (k - args.number_of_features, tm.get_ta_state(j, k, the_class=1, polarity=0)))
        print(" ∧ ".join(l))

    print("\nClass 1 Negative Clauses:\n")

    precision = tm.clause_precision(1, 1, X_test, Y_test)
    recall = tm.clause_recall(1, 1, X_test, Y_test)
 
    for j in range(args.number_of_clauses // 2):
        print("Clause #%d W:%d P:%.2f R:%.2f " % (j, tm.get_weight(1, 1, j), precision[j], recall[j]), end=' ')
        l = []
        for k in range(4):
            if tm.get_ta_action(j, k, the_class=1, polarity=1):
                if k < args.number_of_features:
                    l.append(" x%d(%d)" % (k, tm.get_ta_state(j, k, the_class=1, polarity=1)))
                else:
                    l.append("¬x%d(%d)" % (k - args.number_of_features, tm.get_ta_state(j, k, the_class=1, polarity=1)))
        print(" ∧ ".join(l))

    print("\nClause Co-Occurence Matrix:\n")
    print(tm.clause_co_occurrence(X_test, percentage=True).toarray())

    print("\nLiteral Frequency:\n")
    print(tm.literal_clause_frequency())

    return experiment_results


def default_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=250, type=int)
    parser.add_argument("--number-of-examples", default=10000, type=int)
    parser.add_argument("--number-of-clauses", default=10, type=int)
    parser.add_argument("--platform", default='CPU_sets', type=str)
    parser.add_argument("--T", default=80, type=int)
    parser.add_argument("--s", default=1.0, type=float)
    parser.add_argument("--number-of-features", default=20, type=int)
    parser.add_argument("--noise", default=0.1, type=float, help="Noisy XOR")
    args = parser.parse_args()
    for key, value in kwargs.items():
        if key in args.__dict__:
            setattr(args, key, value)
    return args


if __name__ == "__main__":
    results = main(default_args())
