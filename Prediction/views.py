import os
import numpy as np
import pandas as pd
from rest_framework.response import Response
from rest_framework.views import APIView

from Prediction.serializers import InsuranceClaimSerializer
from .apps import PredictionConfig


def create_records(X, predictors, classifier, classification_threshold):
    class_names = ["자동지급", "심사", "조사"]
    class_prob = classifier.predict_proba(predictors.values.reshape((1,-1)))
    insurance_claim = dict(X.to_dict(), **dict(zip(class_names, np.round(class_prob[0] * 100, 2))))
    insurance_claim["prediction"] = class_names[class_prob.argmax()]
    pred = class_names[class_prob.argmax()]
    insurance_claim["pred"] = pred

    if np.random.randint(0, 100) < 25:

        insurance_claim["target"] = None
        insurance_claim["sampling_method"] = "random"

    elif class_prob.max() > classification_threshold:

        insurance_claim["target"] = pred
        insurance_claim["sampling_method"] = "automation"

    else:

        insurance_claim["target"] = None
        insurance_claim["sampling_method"] = "confidence"

    insurance_claim["conf"] = class_prob.max()


    # 보험금 청구 건 데이터에 클래스 확률 추가해서 저장
    serializer = InsuranceClaimSerializer(data=insurance_claim)
    assert serializer.is_valid(), serializer.errors
    serializer.save()


class InsuranceClaimPredict(APIView):
    def post(self, request):
        data = request.data
        X = pd.Series(data).astype(float)

        if X["prm_nvcd"] == 99:

            classifier = PredictionConfig.classifier_na
            predictors = X.drop(["prm_nvcd", "base_ym", "ID"])

        else:

            classifier = PredictionConfig.classifier_normal
            predictors = X.drop(["base_ym", "ID"])

        try:

            create_records(X, predictors, classifier, float(os.environ["THRESHOLD"]) * 0.01)
            return Response(status=200)

        except AssertionError:

            return Response(status=400)