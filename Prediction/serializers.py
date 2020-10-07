from rest_framework import serializers
from app.models import InsuranceClaim, LimeReport


class InsuranceClaimSerializer(serializers.ModelSerializer):
    class Meta:
        model = InsuranceClaim
        fields = '__all__'


class LimeReportSerializer(serializers.ModelSerializer):
    claim = serializers.PrimaryKeyRelatedField(many=True, read_only=True)
    claim_id = serializers.IntegerField(write_only=True)
    discretized = serializers.JSONField() # change is here

    class Meta:
        model = LimeReport
        fields = '__all__'