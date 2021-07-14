# %%
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import KMeansSMOTE
from imblearn.over_sampling import SVMSMOTE
from imblearn.under_sampling import OneSidedSelection
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    accuracy_score,
)
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from imblearn.over_sampling import ADASYN
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score


# %%
dataset = "iemocap"
root_path = f"/Users/martin/Documents/UNIVERSIDAD/CLASES/4º/2o Cuatri/TFG/data/{dataset}"

iemocap = {
    0: ("ang", "r"),
    1: ("hap", "g"),
    2: ("sad", "b"),
    3: ("neu", "pink"),
    4: ("fru", "m"),
    5: ("exc", "y"),
    6: ("fea", "k"),
    7: ("sur", "c"),
    8: ("dis", "lime"),
    9: ("oth", "tab:grey"),
}

msp = {
    0: ("ang", "r"),
    1: ("hap", "g"),
    2: ("sad", "b"),
    3: ("neu", "pink"),
    4: ("com", "m"),
    5: ("fea", "y"),
    6: ("sur", "k"),
    7: ("dis", "c"),
    8: ("oth", "lime"),
}

if dataset == "iemocap":
    classes = iemocap
else:
    classes = msp


def clean_data(X_train, y_train, X_test, y_test):
    # Cleaning the data
    mapping = {}
    ids = {}
    for k, v in classes.items():
        emo_name = v[0]
        emo_num = k
        mapping[emo_name] = emo_num
    if dataset == "iemocap":
        ids["oth"] = np.argwhere(y_train == mapping["oth"])
        y_train = np.delete(y_train, ids["oth"], 0)
        X_train = np.delete(X_train, ids["oth"], 0)
        ids["oth"] = np.argwhere(y_test == mapping["oth"])
        y_test = np.delete(y_test, ids["oth"], 0)
        X_test = np.delete(X_test, ids["oth"], 0)
        del classes[mapping["oth"]]

        ids["fru"] = np.argwhere(y_train == mapping["fru"])
        y_train = np.delete(y_train, ids["fru"], 0)
        X_train = np.delete(X_train, ids["fru"], 0)
        ids["fru"] = np.argwhere(y_test == mapping["fru"])
        y_test = np.delete(y_test, ids["fru"], 0)
        X_test = np.delete(X_test, ids["fru"], 0)
        del classes[mapping["fru"]]

        ids["dis"] = np.argwhere(y_train == mapping["dis"])
        y_train = np.delete(y_train, ids["dis"], 0)
        X_train = np.delete(X_train, ids["dis"], 0)
        ids["dis"] = np.argwhere(y_test == mapping["dis"])
        y_test = np.delete(y_test, ids["dis"], 0)
        X_test = np.delete(X_test, ids["dis"], 0)
        del classes[mapping["dis"]]

        # merge excitement with happiness
        ids["exc"] = np.argwhere(y_train == mapping["exc"])
        y_train[ids["exc"]] = mapping["hap"]
        X_train[ids["exc"]] = mapping["hap"]
        ids["exc"] = np.argwhere(y_test == mapping["exc"])
        y_test[ids["exc"]] = mapping["hap"]
        X_test[ids["exc"]] = mapping["hap"]
        del classes[mapping["exc"]]

    if dataset == "msp":
        ids["oth"] = np.argwhere(y_train == mapping["oth"])
        y_train = np.delete(y_train, ids["oth"], 0)
        X_train = np.delete(X_train, ids["oth"], 0)
        ids["oth"] = np.argwhere(y_test == mapping["oth"])
        y_test = np.delete(y_test, ids["oth"], 0)
        X_test = np.delete(X_test, ids["oth"], 0)
        del classes[mapping["oth"]]

        ids["com"] = np.argwhere(y_train == mapping["com"])
        y_train = np.delete(y_train, ids["com"], 0)
        X_train = np.delete(X_train, ids["com"], 0)
        ids["com"] = np.argwhere(y_test == mapping["com"])
        y_test = np.delete(y_test, ids["com"], 0)
        X_test = np.delete(X_test, ids["com"], 0)
        del classes[mapping["com"]]

    return X_train, y_train, X_test, y_test


# %%
def dry_run(clf, X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    print(f"Accuracy: {acc}")
    print("-" * 20)

    preds = clf.predict(X_test)
    print(classification_report(y_test, preds))
    print("-" * 20)

    title = ""
    column = ""
    for name in [v[0] for k, v in classes.items()]:
        title += f"{name}\t"
        column += f"{name}"
    confusions = confusion_matrix(y_test, preds)
    print("#########\t Confusion Matrix \t#########")
    print(f"\t{title}")

    for i in range(len(confusions)):
        row = ""
        for c in confusions[i]:
            row += f"{c}\t"
        print(f"^{column[3*i:3*(i+1)]}|\t{row}")


# %%
# Check imbalances in training data
def get_weights(labels):
    weights = {}
    for k, v in classes.items():
        emo_name = v[0]
        weights[emo_name] = 0

    for label in labels:
        weights[label] += 1
    counts = np.asarray([v for k, v in weights.items()])
    most_common = np.max(counts)
    for k, v in weights.items():
        w = most_common / v
        weights[k] = w

    return weights


# %%
def get_counts(samples):
    # counts = { k: len(np.argwhere(samples==k)) for k,v in dic.items()} Not very readable
    counts = {k: 0 for k, v in classes.items()}
    total_count = 0
    for x in samples:
        if x in counts.keys():
            counts[x] = counts[x] + 1
            total_count = total_count + 1

    return counts, total_count


# %%
def plot_balances(emo):
    counts, tot_count = get_counts(emo)

    class_names = [v[0] for k, v in classes.items()]
    colors = [v[1] for k, v in classes.items()]

    class_index = [k for k in counts.keys()]
    counts = [v for v in counts.values()]

    plt.figure(figsize=(15, 8))
    graph = plt.bar(class_index, counts, align="center", color=colors)
    plt.xticks(class_index, class_names)
    plt.ylabel("N Samples")

    i = 0
    for p in graph:
        width = p.get_width()
        height = p.get_height()
        x, y = p.get_xy()
        plt.text(
            x + width / 2,
            y + height * 1.01,
            f"{(counts[i] / tot_count) * 100:.2f}%",
            ha="center",
            weight="bold",
        )
        i += 1
    plt.show()


# %%
def load_data(root_path):
    input = np.load(f"{root_path}/dimension.npy")
    labels = np.load(f"{root_path}/emotions.npy")

    mapping = {}
    ids = {}
    for k, v in classes.items():
        emo_name = v[0]
        emo_num = k
        mapping[emo_name] = emo_num

    ids["nan"] = np.argwhere(np.isnan(labels))
    labels = np.delete(labels, ids["nan"], 0)
    input = np.delete(input, ids["nan"], 0)

    labels = labels.astype(int)

    class_names = []
    for i in range(len(labels)):
        class_names.append(classes[labels[i]][0])

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(input)
    input = scaler.transform(input)

    return input, labels


# %%
# Self developed algorithm based on the literature findings for the emotions centers
class PaperBasedAlgorithm:
    anger = np.array([-0.43, 0.67, 0.34])
    joy = np.array([0.76, 0.48, 0.35])
    surprise = np.array([0.4, 0.67, -0.13])
    neutral = np.array([0.0, 0.0, 0.0])
    disgust = np.array([-0.6, 0.35, 0.11])
    fear = np.array([-0.64, 0.6, -0.43])
    sadness = np.array([-0.63, 0.27, -0.33])

    def __init__(self):
        self.emotions = np.asarray(
            [self.anger, self.joy, self.sadness,
                self.neutral, self.fear, self.surprise]
        )

    def run(self, X, y):
        # print("Predicted\t|\tTruth")
        # for sample, label in zip(X, y):
        #     dist = np.sum((emotions - sample) ** 2, axis=1)
        #     emo_idx = np.argmin(dist)
        #     print(f"{classes[emo_idx]}\t\t{iemocap[label]}")
        preds = []
        for sample in X:
            dist = np.sum((self.emotions - sample) ** 2, axis=1)
            pred = np.argmin(dist)
            preds.append(pred)
        preds = np.asarray(preds)
        return preds

    def fit(self, X, y):
        preds = self.run(X, y)
        return None

    def predict(self, X):
        preds = self.run(X, y)
        return preds

    def score(self, X, y):
        preds = self.run(X, y)
        score = accuracy_score(y, preds)
        return score


# %%
# * Validation
clfs = {
    "RandomForest": RandomForestClassifier(),
    "NaiveBayes": GaussianNB(),
    "XGB": XGBClassifier(),
    "NeuralNetwork-256/512/256": MLPClassifier(
        hidden_layer_sizes=(256, 512, 256), max_iter=1000
    ),
    "NeuralNetwork-256/512/1024/512/256": MLPClassifier(
        hidden_layer_sizes=(256, 512, 1024, 512, 256), max_iter=1000
    ),
    "KNN": KNeighborsClassifier(),
    "SVM-C=5": SVC(C=5.0),
    "SVM-C=1": SVC(C=1.0),
    "SVM-C=10": SVC(C=10.0),
}
params = {
    "RandomForest": None,
    "NaiveBayes": None,
    "XGB": None,
    "NeuralNetwork": None,
    "KNN": None,
    "SVM": None,
}
scoring = {'accuracy': make_scorer(accuracy_score),
           'precision': make_scorer(precision_score, average='weighted'),
           'recall': make_scorer(recall_score,  average='weighted'),
           'f1_score': make_scorer(f1_score,  average='weighted')
           }

X, y = load_data(root_path)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

plot_balances(y)

X_train, y_train, X_test, y_test = clean_data(X_train, y_train, X_test, y_test)
X = np.concatenate((X_train, X_test))
y = np.concatenate((y_train, y_test))
print("#" * 60)
print("\t\tAfter Cleaning Data")
print("#" * 60)
plot_balances(y)
'''
for clf_name, clf in clfs.items():
        # searcher = GridSearchCV(estimator=clf, n_jobs=8, cv=5)
        # print(searcher.cv_results_)
        separator = "-" * 50
        print(separator)
        print(clf_name)
        cv = 10
        clf.fit(X, y)
        print(clf.get_params())
        print("-" * 50)
'''
with open(file="results_classification.txt", mode="w") as f:
    for clf_name, clf in clfs.items():
        # searcher = GridSearchCV(estimator=clf, n_jobs=8, cv=5)
        # print(searcher.cv_results_)
        separator = "-" * 50
        print(separator)
        print(clf_name)
        f.write(f"{separator}\n")
        f.write(f"{clf_name}\n")
        cv = 10
        # scores = cross_val_score(clf, X, y, cv=cv)
        scores = cross_validate(clf, X, y, cv=cv, scoring=scoring)
        for s in ['accuracy', 'precision', 'recall', 'f1_score']:
            s_name = f'test_{s}'
            results = scores[s_name]
            avg = np.sum(results)/cv
            std = np.std(results)
            print(f'Weigthed_{s}: {results}\tAvg:{avg:.4f}\tStd:{std:.4f}')
            f.write(f'Weigthed_{s}: {results}\tAvg:{avg:.4f}\tStd:{std:.4f}\n')
        print("-" * 50)
        f.write(f"{separator}\n")


# ------------------------------------------------------------------------------
#                              EXPERIMENTAL STUFF
# ------------------------------------------------------------------------------

# %%
# clf = XGBClassifier()
# clf = RandomForestClassifier()
# clf = PaperBasedAlgorithm()
# clf = GaussianNB()
# clf = MLPClassifier(hidden_layer_sizes=(256, 512, 256))
# clf = KNeighborsClassifier()
clf = SVC()

# %%
X, y = load_data(root_path)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

print("\t\t\tDATA")
print(f"Input|\tTrain: {X_train.shape}\tTest: {X_test.shape}")
print(f"Labels|\tTrain: {y_train.shape}\tTest: {y_test.shape}")

plot_balances(y_train)

dry_run(clf, X_train, X_test, y_train, y_test)

# %%
X_train, y_train, X_test, y_test = clean_data(X_train, y_train, X_test, y_test)
print("#" * 60)
print("\t\tAfter Cleaning Data")
print("#" * 60)
plot_balances(y_train)
dry_run(clf, X_train, X_test, y_train, y_test)

# %%
# ? Undersampling

if dataset == "iemocap":
    # define the undersampling method
    undersample = OneSidedSelection(
        "not_minority", n_neighbors=1, n_seeds_S=1000)
    # transform the dataset
    X_train, y_train = undersample.fit_resample(X_train, y_train)
if dataset == "msp":
    # define the undersampling method
    undersample = OneSidedSelection(
        sampling_strategy="not minority", n_neighbors=1, n_seeds_S=2000
    )
    # transform the dataset
    X_train, y_train = undersample.fit_resample(X_train, y_train)

    undersample = OneSidedSelection(
        sampling_strategy=[mapping["neu"]], n_neighbors=1, n_seeds_S=2000
    )


print("#" * 60)
print("\t\tAfter Undersampling Majority Classes")
print("#" * 60)

plot_balances(y_train)
dry_run(clf, X_train, X_test, y_train, y_test)


# %%
# ? Oversampling


if dataset == "iemocap":
    # oversample = SVMSMOTE(sampling
    # oversample = SVMSMOTE(sampling_strategy={6: 1000, 7:1000})
    oversample = ADASYN(sampling_strategy={6: 1000, 7: 1000})
    X, y = oversample.fit_resample(X, y)

if dataset == "msp":
    oversample = KMeansSMOTE()
    X_train, y_train = oversample.fit_resample(X_train, y_train)

print("#" * 60)
print("\t\tAfter Oversampling Minority Classes")
print("#" * 60)

plot_balances(y_train)
dry_run(clf, X_train, X_test, y_train, y_test)
