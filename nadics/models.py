###############################################################################
#                                                                             #
# Here are all models listed we provide with either binary-, '2', or multi-   #
# classification purpose, 'n'.                                                #
# We used a Grid Search to determine good hyperparamters for our models and   #
# set them afterwards for our needs.                                          #
#                                                                             #
# NOTE: Feel free to add other models and be aware of their fitting methods.  #
# E.g. some model for outliers detection might be called differently than     #
# other and thus, might need some modifications.                              #
#                                                                             #
###############################################################################

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.svm import NuSVC
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import CondensedNearestNeighbour
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import RepeatedEditedNearestNeighbours
from imblearn.under_sampling import AllKNN
from imblearn.under_sampling import InstanceHardnessThreshold
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import NeighbourhoodCleaningRule
from imblearn.under_sampling import OneSidedSelection
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek
from imblearn.combine import SMOTEENN
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope


classifications = ["2", "n"]

classifiers = {
    "AdaBoost": AdaBoostClassifier(),
    "DecisionTree": DecisionTreeClassifier(),
    "RandomForest": RandomForestClassifier(
        n_estimators=200,
        criterion="gini",
        max_features="auto",
        max_depth=None,
        max_leaf_nodes=None,
        n_jobs=-1),
    "KNeighbors": KNeighborsClassifier(
        n_jobs=-1),
    "GaussianProcess": GaussianProcessClassifier(
        n_jobs=-1),
    "GaussianNB": GaussianNB(),
    "BernoulliNB": BernoulliNB(),
    "MultinomialNB": MultinomialNB(),
    "MLP": MLPClassifier(),
    "SGD": SGDClassifier(
        n_jobs=-1),
    "C_SVC": SVC(
        cache_size=2048),
    "Linear_SVC": LinearSVC(),
    "Nu_SVC": NuSVC(
        cache_size=2048)
    }

clustering = {
    "IsolationForest": IsolationForest(),
    "OSVM": OneClassSVM(),
    "EllipticEnvelope": EllipticEnvelope(),
    "LOF": LocalOutlierFactor()
    }

samplings = {
    "ADASYN": ADASYN(),
    "RandomOverSampler": RandomOverSampler(),
    "SMOTE": SMOTE(),
    "CondensedNearestNeighbour": CondensedNearestNeighbour(),
    "EditedNN": EditedNearestNeighbours(),
    "RepeatedEditedNN": RepeatedEditedNearestNeighbours(),
    "AllKNN": AllKNN(),
    "InstanceHardnessThreshold": InstanceHardnessThreshold(),
    "NearMiss": NearMiss(),
    "NeighbourhoodCleaningRule": NeighbourhoodCleaningRule(),
    "OneSidedSelection": OneSidedSelection(),
    "RandomUnderSampler": RandomUnderSampler(),
    "TomekLinks": TomekLinks(),
    "SMOTETomek": SMOTETomek(),
    "SMOTEENN": SMOTEENN()
    }

boostings = {
    "GradientBoosting": GradientBoostingClassifier(),
    "AdaBoost": AdaBoostClassifier(
        algorithm='SAMME')
    }
