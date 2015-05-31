import re
import math
import random
from operator import itemgetter

import os
import numpy as np
from django.conf import settings
import pandas as pd
from pandas import DataFrame, Series
from sklearn.pipeline import Pipeline
from textpredictions.text_model import TextModel
from nltk import word_tokenize
import json

data_folder = os.path.join(settings.STATIC_ROOT, "textpredictions/")


class DisplayTextModel(TextModel):
    is_dummy_outcome = False
    number_of_observations = None

    def __init__(self, outcomes, texts, parameters_display, verbose=False):

        options = {"lowercase": True, "lemmatize": True, "remove-stopwords": True}
        super(DisplayTextModel, self).__init__(outcomes, texts, 'bag-of-words', options)

        data = DataFrame({"y": outcomes, "text": texts})

        # Storing whether the outcome is a dummy:
        if set(data.y) == set([0, 1]):
            self.is_dummy_outcome = True

        N = data.shape[0]
        self.number_of_observations = N

        data.index = [str(x) for x in range(N)]
        data['y_hat'] = self.pipe.predict(texts)

        ridge = self.pipe.named_steps['ridge_model']

        self.std_X = ridge.std_X
        self.parameters_display = parameters_display
        self.mean_outcome_in_groups = mean_outcome_in_groups(data.y, data.y_hat)
        self.percent_correct = share_correct(data.y, data.y_hat, verbose=verbose)
        self.outcome_summary = get_summary(outcomes)

        self.coef = ridge.coef_
        self.number_of_features = len(self.coef)
        features = self.pipe.named_steps['featurizer'].get_feature_names()
        self.features = [f.split("__")[1] for f in features]

    def get_regression_table(self):
        regression_table = DataFrame({"beta": self.coef, "std_X": self.std_X})
        regression_table.index = self.features
        regression_table['beta_normalized'] = regression_table.beta * regression_table.std_X
        regression_table['effect'] = np.fabs(regression_table['beta_normalized'])
        regression_table = regression_table.sort_index(by='effect', ascending=False)
        return regression_table

    def set_performance(self, outcomes_test, texts, number_sampled=40, verbose=False):
        y_hat_test = self.pipe.predict(texts)
        self.mean_outcome_in_groups = mean_outcome_in_groups(outcomes_test, y_hat_test)
        self.share_correct = share_correct(outcomes_test, y_hat_test, verbose=verbose)
        self.share_correct_print = round(self.share_correct, 3)
        self.texts_test_sample = get_texts_sampled(texts, number_sampled)
        self.texts_test_performance = self.get_texts_test_performance()

    def get_texts_test_performance(self):
        texts_test_performance = []
        # Create probabilities for sample
        performance_examples = []
        i = 0
        for text in self.texts_test_sample:
            predicted_value = self.predict(text)
            texts_test_performance.append((text, predicted_value, i))
            i += 1
        texts_test_performance = sorted(texts_test_performance, key=itemgetter(1))
        return texts_test_performance

    def get_features(self, text):
        """This takes a text and translates it into features. This function thus describes central logic of the TextModel"""
        x = Pipeline(self.pipe.steps[0:-1]).transform([text]).toarray()[0]
        features = Series(self.features)[Series(x) >= 1]
        tab = self.get_regression_table()
        coefficients_for_features = tab[tab.index.isin(features)]
        return coefficients_for_features

    def prediction_summary(self, text):
        summary = {}
        try:
            text = text.encode('utf-8', 'ignore')
        except UnicodeEncodeError:
            pass
        summary["predicted_value"] = self.predict(text)
        tab = self.get_regression_table()
        tab_found = self.get_features(text)
        features_plus_minus = get_features_plus_minus(tab_found)
        summary["important_features_good_and_bad"] = features_plus_minus
        return summary


def get_texts_sampled(texts, number):
    """Returns a sample of size `number` from `texts`"""
    number_of_test_texts = min(number, len(texts))
    return random.sample(texts, number_of_test_texts)


def get_features_plus_minus(tab_found):
    betas = tab_found.transpose().to_dict()
    betas_list_list = [(k, v) for (k, v) in betas.iteritems()]
    betas_minus = ["%s (%s)" % (k, -round(v['beta'], 3)) for (k, v) in betas_list_list if v['beta'] < 0]
    betas_plus = ["%s (%s)" % (k, round(v['beta'], 3)) for (k, v) in betas_list_list if v['beta'] > 0]
    return [betas_plus, betas_minus]


def get_printable_dataframe(data):
    """This is a convenience function for django, which returns a list of lists
    This list of lists turns the index into the first column"""
    data = np.round(data, 3)
    return list(data.itertuples())


def get_summary(x):
    """Creates basic summary statistics for `x`."""
    x = pd.Series(x)
    summ = {}
    summ["min"] = round(np.min(x), 3)
    summ["max"] = round(np.max(x), 3)
    summ["mean"] = round(np.mean(x), 3)
    summ["median"] = round(x.quantile(), 3)
    summ["sd"] = round(np.std(x), 3)
    return summ


def get_cutoffs(x, num_groups=10):
    """Get the cutoffs that splits `x` into `num_groups` equally sized groups."""
    series = Series(x)
    cutoffs = []
    for i in range(num_groups):
        perc_low = float(i) / num_groups
        perc_high = float(i + 1) / num_groups
        cutoffs.append((series.quantile(perc_low), series.quantile(perc_high)))
    return cutoffs


def share_correct(y, y_hat, verbose=False):
    """This function is only relevant for binary models. For these models, it shows the percentage of predictions
    correctly classified using the prediction y_hat. This assumes that we classify y=1 if y_hat>.5 and y=0 otherwise"""
    df = pd.DataFrame({"y": y, "y_hat": y_hat})
    df["y_classifier"] = (df.y_hat > .5)
    df["correctly_classified"] = df.y_classifier == df.y
    return df.correctly_classified.mean()


def mean_outcome_in_groups(y, y_hat, num_groups=10, verbose=False):
    """Get the average of the outcome y when y_hat is cut into num_groups equally-sized groups. This 
    is used as a measure of performance of the model"""
    cutoffs = get_cutoffs(y_hat, num_groups)
    return mean_outcome_by_cutoff(y, y_hat, cutoffs)


def mean_outcome_by_cutoff(y, y_hat, cutoffs, verbose=False):
    """Show the average outcome y by the cutoffs for y_hat"""
    y_by_group = []
    df = pd.DataFrame({"y": y, "y_hat": y_hat})
    # Get performance from test sample (test==2), not from valdiation sample (test==1)
    for cutoff_low, cutoff_high in cutoffs:
        data_group = df[(df.y_hat >= cutoff_low) & (df.y_hat < cutoff_high)]
        y_by_group.append(np.mean(data_group["y"]))
    performance = []
    for i in range(len(cutoffs)):
        performance.append((i + 1, round(y_by_group[i], 3)))
    return performance


def convert_performance_to_string(performance):
    performance_string = []
    note = ""
    for decile in performance:
        value = decile[1]
        value_string = str(value)
        if math.isnan(value):
            value_string = "N/A*"
            note = "Since deciles are determined using the training sample, it cannot be ensured that all deciles can be evaluated in the test sample"
        decile_string = (decile[0], value_string)
        performance_string.append(decile_string)
    return (performance_string, note)


def text_model_parameters(filename, train=True, verbose=False):
    train_string = "_train"
    if not train:
        train_string = "_test"

    filename_full = data_folder + "/" + filename + train_string + ".csv"

    descriptions_filename = "textpredictions/static/textpredictions/descriptions.json"

    descriptions = json.load(open(descriptions_filename, "r"))

    display_parameters = descriptions[filename]

    data_original = pd.read_csv(filename_full)

    outcome = display_parameters['outcome']
    text_variable = display_parameters['text_variable']

    if not outcome:
        raise (ValueError("No outcome set"))
    if not text_variable:
        raise (ValueError("No text variable set"))

    data = data_original[[outcome, text_variable]]
    data.columns = ["y", "text"]
    outcomes = data.y
    texts = data.text
    return outcomes, outcome, texts, text_variable, display_parameters


def get_similarity(text_1, text_2):
    words_1 = set(word_tokenize(text_1.lower()))
    words_2 = set(word_tokenize(text_2.lower()))
    if len(words_1) == 0 or len(words_2) == 0:
        return 0, 0

    intersection = words_1.intersection(words_2)
    return (float(len(intersection)) / float(len(words_1)), float(len(intersection)) / float(len(words_2)))
