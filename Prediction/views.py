import random
import pandas as pd
from numpy import round
from rest_framework.response import Response
from rest_framework.views import APIView

from Prediction.serializers import InsuranceClaimSerializer, LimeReportSerializer
from .apps import PredictionConfig


def create_records(explanation, X, predictors, classification_threshold=0.9):
    label = explanation.predict_proba.argmax()
    local_exp = explanation.local_exp.get(label)
    listed = explanation.as_list(label=label)
    class_prob = dict(zip(explanation.class_names, round(explanation.predict_proba * 100, 2)))

    # 보험금 청구 건 데이터에 클래스 확률 추가해서 저장
    insurance_claim = dict(X.to_dict(), **class_prob)
    if explanation.predict_proba.max() > classification_threshold:
        insurance_claim["target"] = explanation.class_names[explanation.predict_proba.argmax()]
    serializer = InsuranceClaimSerializer(data=insurance_claim)
    assert serializer.is_valid(), serializer.errors
    serializer.save()

    # 라임 분석 데이터 저장
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
    serializer = LimeReportSerializer(data=lime_report, many=True)
    assert serializer.is_valid(), serializer.errors
    serializer.save()
    return 200


class InsuranceClaimPredict(APIView):
    def post(self, request, format=None):
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
        random.seed(0)
        explanation = explainer.explain_instance(predictors, classifier.predict_proba, num_features=3, top_labels=3)
        response_code = create_records(explanation, X, predictors)
        return Response(status=response_code)
