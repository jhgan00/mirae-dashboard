import os
import numpy as np
import pandas as pd
from rest_framework.response import Response
from rest_framework.views import APIView

from Prediction.serializers import InsuranceClaimSerializer, LimeReportSerializer
from .apps import PredictionConfig


def create_records(X, predictors, classifier, explainer, classification_threshold):
    class_names = ["자동지급", "심사", "조사"]
    class_prob = classifier.predict_proba(predictors.values.reshape((1,-1)))
    insurance_claim = dict(X.to_dict(), **dict(zip(class_names, np.round(class_prob[0] * 100, 2))))
    insurance_claim["prediction"] = class_names[class_prob.argmax()]

    if class_prob.max() > classification_threshold:

        insurance_claim["target"] = class_names[class_prob.argmax()]

    else:

        insurance_claim["target"] = None
        np.random.seed(0)
        explanation = explainer.explain_instance(
            predictors, classifier.predict_proba, num_features=3, top_labels=1, labels=[0, 1, 2]
        )
        label = explanation.predict_proba.argmax()
        local_exp = explanation.local_exp.get(label)
        listed = explanation.as_list(label=label)
        claim_id = int(X.loc["ID"])
        lime_report = [dict(
            claim=[claim_id],
            claim_id=claim_id,
            rank=i,
            feature=predictors.index[local_exp[i][0]],
            local_exp=local_exp[i][1],
            discretized=listed[i][0],
            value=predictors.iloc[local_exp[i][0]]
        ) for i in range(3)]

    # 보험금 청구 건 데이터에 클래스 확률 추가해서 저장
    serializer = InsuranceClaimSerializer(data=insurance_claim)
    assert serializer.is_valid(), serializer.errors
    serializer.save()

    # 라임 분석 데이터 저장
    if 'lime_report' in locals():

        serializer = LimeReportSerializer(data=lime_report, many=True)
        assert serializer.is_valid(), serializer.errors
        serializer.save()


class InsuranceClaimPredict(APIView):
    def post(self, request):
        data = request.data
        X = pd.Series(data).astype(float)

        if X["prm_nvcd"] == 99:

            classifier = PredictionConfig.classifier_na
            explainer = PredictionConfig.explainer_na
            predictors = X.drop(["prm_nvcd", "base_ym", "ID"])

        else:

            classifier = PredictionConfig.classifier_normal
            explainer = PredictionConfig.explainer_normal
            predictors = X.drop(["base_ym", "ID"])

        try:

            create_records(X, predictors, classifier, explainer, float(os.environ["THRESHOLD"]) * 0.01)
            return Response(status=200)

        except AssertionError:

            return Response(status=400)