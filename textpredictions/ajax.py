import json

from dajaxice.decorators import dajaxice_register
from textpredictions.models import PredictionModel


@dajaxice_register
def get_results(request, text, model_id):
    model = PredictionModel.objects.get(pk=model_id)
    text_model = model.text_model
    prediction_summ = text_model.prediction_summary(text)
    message = 'Text: %s | Id: %s | Predicted: %s' % (text, model_id, prediction_summ['predicted_value'])
    return json.dumps({'prediction_summ': prediction_summ, 'message': message})


