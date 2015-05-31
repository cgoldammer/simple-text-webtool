from django.shortcuts import render
from django.views import generic
from textpredictions.models import PredictionModel, TextEntryForm


def about(request):
    return render(request, 'textpredictions/about.html', )


def enter_text(request, model_pk, text_entry_pk):
    print "Going to text entry"
    print text_entry_pk
    print model_pk
    if text_entry_pk:
        print "Model or text entry received"
        # textEntry=TextEntry.objects.get(pk=text_entry_pk)
        model = PredictionModel.objects.get(pk=model_pk)
        text = model.text_model.texts_test_sample[int(text_entry_pk)]
        return render(request, 'textpredictions/enter_text.html', {'text': text, "model": model})
    if model_pk:
        model = PredictionModel.objects.get(pk=model_pk)
        return render(request, 'textpredictions/enter_text.html', {"model": model})
    else:
        form = TextEntryForm()
    return render(request, 'textpredictions/enter_text.html', {'form': form})


def model(request, model_pk):
    model = PredictionModel.objects.get(pk=model_pk)
    return render(request, 'textpredictions/model.html', {"model": model})


class TechnicalModelView(generic.ListView):
    template_name = 'textpredictions/models_technical.html'
    context_object_name = 'models'

    def get_queryset(self):
        """Return all models."""
        return PredictionModel.objects.all()


class ModelView(generic.ListView):
    template_name = 'textpredictions/list_of_models.html'
    context_object_name = 'models'

    def get_queryset(self):
        """Return all models."""
        return PredictionModel.objects.all()


from django.shortcuts import render

# import logging

def index(request):
    return render(request, 'libs/index.html', {})