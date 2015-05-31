import os
import urllib
import json
import unittest

from textpredictions.models import PredictionModel
from django.test import TestCase
from django.core.urlresolvers import reverse
from textpredictions import text_model_functions
from django.conf import settings
from django.test import Client


def create_one_model():
    filename = "obama_or_lincoln"
    (outcomes_train, outcome_varname, texts_train, text_varname,
     parameters_display) = text_model_functions.text_model_parameters(filename=filename, train=True)
    text_model = text_model_functions.DisplayTextModel(outcomes_train, texts_train, parameters_display=parameters_display)

    # Set performance using test sample
    (outcomes_test, outcomes_varname, texts_test, texts_varname,
     parameters_display) = text_model_functions.text_model_parameters(filename=filename)
    text_model.set_performance(outcomes_test, texts_test)

    prediction_model = PredictionModel(outcome=outcome_varname,
                                       text_variable=text_varname,
                                       model_name=filename,
                                       data_name=filename,
                                       text_model=text_model)
    prediction_model.save()


# Checking that the player works correctly
class TestTextprediction(TestCase):
    folder = os.path.join(settings.STATIC_ROOT, "textpredictions/")

    def setUp(self):
        """Creating the Obama or Lincoln model and saving it"""
        create_one_model()

    def empty(self):
        pass

    def test_parameters(self):
        """The model has more than a 100 observations and exactly 5000 features"""
        prediction_model = PredictionModel.objects.all()[0]
        text_model = prediction_model.text_model
        self.assertTrue(text_model.number_of_observations > 100)

    def test_model_list(self):
        """The model list correctly displays its description"""
        response = self.client.get(reverse('models'))
        self.assertEqual(response.status_code, 200, "Models list returned error")
        response_str = str(response)

        if not "This model uses speeches by Obama and Lincoln" in response_str:
            raise (ValueError, "Model text description not found!")

    def test_model_building(self):
        filenames = ["obama_or_lincoln"]  # "subjective_or_objective"
        for filename in filenames:

            (outcomes_train, outcome_varname, texts_train, text_varname,
             parameters_display) = text_model_functions.text_model_parameters(filename=filename, train=True)
            text_model = text_model_functions.DisplayTextModel(outcomes_train, texts_train,
                                                        parameters_display=parameters_display)

            # Set performance using test sample
            (outcomes_test, outcomes_varname, texts_test, texts_varname,
             parameters_display) = text_model_functions.text_model_parameters(filename=filename, train=False)
            text_model.set_performance(outcomes_test, texts_test)

            prediction_model = PredictionModel(outcome=outcome_varname,
                                               text_variable=text_varname,
                                               model_name=filename,
                                               data_name=filename,
                                               text_model=text_model)

            # Save the model
            prediction_model.save()

            # Make sure that the overview page works
            prediction_model_id = prediction_model.id
            self.assertTrue(prediction_model_id > 0)

            response = self.client.get(reverse('model', args=(prediction_model_id,)))
            self.assertEqual(response.status_code, 200)

            # Test that the page contains both a summary and example observations
            if not "Share correctly predicted" in str(response):
                raise (ValueError, "Model page does not display the share correctly predicted")

            response_str = str(response)
            if not "Example observations" in response_str:
                sample = prediction_model.text_model.get_texts_test_performance()
                raise (ValueError, "Model page does not display example observations")

            if filename == "obama_or_lincoln":
                if not "Obama or Lincoln?" in response_str:
                    raise (ValueError)
                if not "This model uses speeches by Obama and Lincoln" in response_str:
                    raise (ValueError)


class TestAjax(TestCase):
    def setUp(self):
        """Creating the Obama or Lincoln model and saving it"""
        create_one_model()

    """The Ajax calls works as expected"""
    url = "/dajaxice/textpredictions.get_results/"

    def test_results(self):
        c = Client()
        prediction_model = PredictionModel.objects.all()[0]
        text = "McCain"
        content = {"text": text, "model_id": prediction_model.id}
        data = {"argv": json.dumps(content)}
        response = self.client.post(self.url,
                                    data=urllib.urlencode(data),
                                    content_type='application/x-www-form-urlencoded',
                                    HTTP_X_REQUESTED_WITH='XMLHttpRequest')
        self.assertEqual(response.status_code, 200)

        results = json.loads(response.content)
        summ = results["prediction_summ"]
        self.assertTrue(summ["predicted_value"] > 0.55)

        feats = summ['important_features_good_and_bad']
        self.assertTrue('mccain' in feats[0][0])


class TestModel(unittest.TestCase):
    def setUp(self):
        filename = "obama_or_lincoln"
        (outcomes_train, outcome_varname, texts_train, text_varname,
         parameters_display) = text_model_functions.text_model_parameters(filename=filename, train=True)
        self.outcomes = outcomes_train
        self.texts = texts_train
        self.text_model = text_model_functions.DisplayTextModel(outcomes_train, texts_train, parameters_display)

    def test_std(self):
        self.assertTrue(len(self.text_model.std_X) > 0)

    def test_coef(self):
        self.assertTrue(len(self.text_model.coef) > 0)

    def test_table(self):
        """Making sure that the coefficients are correctly displayed"""
        table = self.text_model.get_regression_table()
        self.assertTrue("__" not in table.index[0])

    def test_performance(self):
        tm = self.text_model
        tm.set_performance(self.outcomes, self.texts)
        self.text_model.get_texts_test_performance()

    def test_summary(self):
        pass
        #self.text_model.prediction_summary("McCain")

    def test_prediction(self):
        self.assertTrue(self.text_model.predict("McCain") > 0.55)

class TestHelperFunctions(unittest.TestCase):
    def test_parameters(self):
        folder = os.path.join(settings.STATIC_ROOT, "textpredictions/")
        filename = "obama_or_lincoln"
        (outcomes_train, outcome_varname, texts_train, text_varname,
         parameters_display) = text_model_functions.text_model_parameters(filename=filename, train=True, verbose=False)
        self.assertEquals(parameters_display['model_name'], "Obama or Lincoln?")






