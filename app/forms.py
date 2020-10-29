from django import forms
from app.models import InsuranceClaim


class CostForm(forms.Form):
    cost01 = forms.FloatField()
    cost02 = forms.FloatField()
    cost10 = forms.FloatField()
    cost12 = forms.FloatField()
    cost20 = forms.FloatField()
    cost21 = forms.FloatField()
    manual = forms.FloatField()


class InsuranceClaimForm(forms.ModelForm):
    class Meta:
        model = InsuranceClaim
        exclude = ['자동지급','심사','조사','target','conf','pred','sampling_method']


class InsuranceClaimUpdateForm(forms.ModelForm):
    target = forms.ChoiceField(choices=[("자동지급", "자동지급"),("심사","심사"),("조사","조사")], widget=forms.RadioSelect(attrs={'class': 'select'}))

    class Meta:
        model = InsuranceClaim
        fields = ["target"]
#
# class ChildModel(ModelForm):
#     secretdocs = forms.ChoiceField(choices=[(doc.uid, doc.name) for doc in Document.objects.all()])
#
#     class Meta:
#         model = Documents
#         fields = ('secretdocs',)
#         widgets = {
#             'secretdocs': forms.Select(attrs={'class': 'select'}),
#         }