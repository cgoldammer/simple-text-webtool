import os

from django.core.management.base import NoArgsCommand
from textpredictions.models import PredictionModel
from textpredictions import text_model_functions
from django.conf import settings


class Command(NoArgsCommand):
    help = 'Adds models for csvs in staticfiles'
    folder = os.path.join(settings.STATIC_ROOT, "textpredictions/")

    def handle_noargs(self, **options):
        PredictionModel.objects.all().delete()
        filenames = ["obama_or_lincoln"]  # , "movie_reviews","subjective_or_objective"]  # ,
        for filename in filenames:
            print "Testing creating model from file: %s" % (filename)

            (outcomes_train, outcome_varname, texts_train, text_varname,
             parameters_display) = text_model_functions.text_model_parameters(filename=filename, train=True)
            text_model = text_model_functions.DisplayTextModel(outcomes_train, texts_train, parameters_display)

            # Set performance using test sample
            (outcomes_test, outcomes_varname, texts_test, texts_varname,
             parameters_display) = text_model_functions.text_model_parameters(filename=filename, train=False)
            text_model.set_performance(outcomes_test, texts_test)

            prediction_model = PredictionModel(outcome=outcome_varname,
                                               text_variable=text_varname,
                                               model_name=filename,
                                               data_name=filename,
                                               text_model=text_model)

            prediction_model.save()
        


