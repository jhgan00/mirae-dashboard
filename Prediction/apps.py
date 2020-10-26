from django.apps import AppConfig
from dill import load
import os


class PredictionConfig(AppConfig):
    name = 'Prediction'
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # classifier
    CLASSIFIER_DIR = os.path.join(BASE_DIR, "Prediction/classifier", os.environ['base_ym'])
    CLASSIFIER_NORMAL = os.path.join(CLASSIFIER_DIR, "normal")
    CLASSIFIER_NA = os.path.join(CLASSIFIER_DIR, "na")

    with open(CLASSIFIER_NORMAL, "rb") as clf:
        classifier_normal = load(clf)
    with open(CLASSIFIER_NA, "rb") as clf:
        classifier_na = load(clf)

    # explainer
    EXPLAINER_DIR = os.path.join(BASE_DIR, "Prediction/explainer/", os.environ['base_ym'])
    EXPLAINER_NORMAL = os.path.join(EXPLAINER_DIR, "normal")
    EXPLAINER_NA = os.path.join(EXPLAINER_DIR, "na")
    with open(EXPLAINER_NORMAL, "rb") as exp:
        explainer_normal = load(exp)
    with open(EXPLAINER_NA, "rb") as exp:
        explainer_na = load(exp)