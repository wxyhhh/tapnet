from utils import load_ts_data, accuracy
from sklearn.svm import SVC
import numpy as np

# from matplotlib.colors import ListedColormap
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

adj, features, labels, idx_train, idx_val, idx_test = load_ts_data(tensor_format=False)

feat_train, label_train = features[idx_train, :].toarray(), np.argmax(labels[idx_train, :], axis=1)
feat_test, label_test = features[idx_test, :].toarray(), np.argmax(labels[idx_test, :], axis=1)


# clf = MLPClassifier(solver='adam', alpha=1e-5,
#                     hidden_layer_sizes=(5, 2), random_state=1)
clf = MLPClassifier(solver='adam', alpha=1e-5,
                    hidden_layer_sizes=(10), random_state=1)
clf.fit(feat_train, label_train)
score = clf.score(feat_test, label_test)
print("NN: ", score)

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

for clf in classifiers:
    clf.fit(feat_train, label_train)
    score = clf.score(feat_test, label_test)
    print(type(clf).__name__, score)