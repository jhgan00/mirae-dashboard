from django import forms


class CostForm(forms.Form):
    cost01 = forms.FloatField()
    cost02 = forms.FloatField()
    cost10 = forms.FloatField()
    cost12 = forms.FloatField()
    cost20 = forms.FloatField()
    cost21 = forms.FloatField()
    manual = forms.FloatField()