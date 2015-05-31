from django.db import models
from django import forms
from django.forms import ModelForm
from picklefield.fields import PickledObjectField
from textpredictions import text_model_functions


class UploadFileForm(forms.Form):
    model_name = forms.CharField(max_length=50, required=False)
    file = forms.FileField(required=False)

    def __init__(self, *args, **kwargs):
        outcome_choices = kwargs.pop("outcome_choices")
        text_choices = kwargs.pop("text_choices")
        super(UploadFileForm, self).__init__(*args, **kwargs)
        self.fields["outcome"] = forms.ChoiceField(choices=outcome_choices, required=False)
        self.fields["text"] = forms.ChoiceField(choices=text_choices, required=False)
        self.fields["description"] = forms.CharField(max_length=1000, required=False)


class PredictionModel(models.Model):
    model_name = models.CharField(max_length=50)
    description = models.CharField(max_length=1000, null=True, blank=True)
    universe = models.CharField(max_length=1000, null=True, blank=True)
    outcome = models.CharField(max_length=50)
    text_variable = models.CharField(max_length=50)
    data_name = models.CharField(max_length=50)
    text_model = PickledObjectField(null=True, blank=True, protocol=1)

    def most_important_features(self):
        reg = text_model_functions.get_printable_dataframe(self.text_model.regression_table)[:200]
        return reg

    def parameters_display(self):
        return self.text_model.parameters_display

    def __unicode__(self):
        return self.model_name


class TextEntry(models.Model):
    text = models.CharField(max_length=20000, blank=True, null=True)
    time = models.DateTimeField(auto_now_add=True, blank=True, null=True)
    model = models.ForeignKey(PredictionModel, blank=True, null=True)
    is_human = models.BooleanField(blank=True, default=True)

    def __unicode__(self):
        desc = u"id: %s | Model: %s | Time: %s | First ten letters: %s" % (
        str(self.id), self.model.model_name, self.time, self.text[0:10])
        return desc


class TextEntryForm(ModelForm):
    class Meta:
        model = TextEntry
        widgets = {
            'text': forms.Textarea(attrs={'rows': 10, 'cols': 50}),
        }
        fields = ['text']


   
