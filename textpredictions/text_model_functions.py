import re
import math
import random
from operator import itemgetter

import numpy as np

import pandas as pd
from pandas import DataFrame, Series
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
import nltk
from textpredictions import text_model
from nltk import word_tokenize


def get_texts_sampled(texts, number):
    number_of_test_texts = min(number, len(texts))
    texts = random.sample(texts, number_of_test_texts)
    return texts


def get_features_plus_minus(tab_found):
    print tab_found
    betas = tab_found.transpose().to_dict()
    betas_list_list = [(k, v) for (k, v) in betas.iteritems()]
    betas_minus = ["%s (%s)" % (k, -round(v['beta'], 3)) for (k, v) in betas_list_list if v['beta'] < 0]
    betas_plus = ["%s (%s)" % (k, round(v['beta'], 3)) for (k, v) in betas_list_list if v['beta'] > 0]
    return [betas_plus, betas_minus]


def get_printable_dataframe(data):
    """This is a convenience function for django, which returns a list of lists
    This list of lists turns the index into the first column"""
    data = np.round(data, 3)
    list_data = list(data.itertuples())
    return list_data


def get_summary(x):
    x = pd.Series(x)
    summ = {}
    summ["min"] = round(np.min(x), 3)
    summ["max"] = round(np.max(x), 3)
    summ["mean"] = round(np.mean(x), 3)
    summ["median"] = round(x.quantile(), 3)
    summ["sd"] = round(np.std(x), 3)
    return summ


def get_cutoffs(x, num_groups=10):
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


def text_model_parameters(filename, folder, train=True, verbose=False):
    train_string = "_train"
    if not train:
        train_string = "_test"
    data_file_string = open(folder + filename + train_string + ".csv", "r").read()
    content = get_content(data_file_string)

    outcome = None
    text_variable = None
    data_original = content['data']

    outcome_name = None
    text_variable_name = None
    example_texts = None
    description = None
    outcomes = None
    images = None
    known_phrases = None
    prediction_name = None
    image_note = None

    if filename == "obama_or_lincoln":
        model_name = "Obama or Lincoln?"
        outcome = "is_obama"
        outcome_name = "Is Obama"
        prediction_name = "Probability that it's Obama"
        outcomes = ["Lincoln", "Obama"]
        images = ["lincoln", "obama"]
        text_variable = "text"
        text_variable_name = "Sentence said in a speech"
        description = "This model uses speeches by Obama and Lincoln to identify who of the two spoke a sentence."
        example_texts = [
            "We the people are the rightful masters of both Congress and the courts, not to overthrow the Constitution but to overthrow the men who pervert the Constitution.",
            "There is not a liberal America and a conservative America. There is the United States of America."]
        known_phrases = [("Four score and seven years ago", 0)]

    if filename == "movie_reviews":
        model_name = "Is the movie good or bad?"
        outcome = "is_positive"
        outcome_name = "Is positive"
        prediction_name = "Probability that the movie was positively reviewed"
        text_variable = "text"
        text_variable_name = "Sentence from movie review"
        description = "This model combines single sentences from movie reviews with ratings (positive or negative) to predict whether a movie will be good."
        example_texts = ["This movie was great! I loved the acting, especially by John Travolta.",
                         "This was horrible. I just can't describe how disappointed I was that the plot was so shallow."]

    if filename == "subjective_or_objective":
        model_name = "Subjective or objective?"
        outcome = "is_subjective"
        outcome_name = "Is subjective"
        prediction_name = "Probability that it this sentence was subjective (and not objective)"
        text_variable = "text"
        text_variable_name = "Sentence from a movie review or plot"
        description = "This model uses single sentences from movie reviews (subjective) and movie plots (objective) to distinguish whether a sentence is subjective or objective."
        example_texts = ["I liked it, it was very exciting!",
                         "A petty thief is tired of the criminal life and decides to become - a towel! With disaster impending his life is hanging by a thread."]

    display_parameters = {'model_name': model_name,
                          'outcome_name': outcome_name,
                          'prediction_name': prediction_name,
                          'text_variable_name': text_variable_name,
                          'example_texts': example_texts,
                          'description': description,
                          'outcomes': outcomes,
                          'images': images,
                          'known_phrases': known_phrases}

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


class RidgeWithStats(Ridge):
    def fit(self, X, y, sample_weight=1.0):
        self.std_X = DataFrame(X.toarray()).std()
        return Ridge.fit(self, X, y, sample_weight)


class TextModel:
    parameters_display = None
    test_sample_performance = None
    regression_table = None
    is_dummy_outcome = False
    use_known_phrases = False
    number_of_observations = None
    number_of_features = None

    def __init__(self, outcomes, texts, parameters_initial={}, parameters_display={}, metadata=None, verbose=False):

        options = {"lowercase": True, "lemmatize": True, "remove-stopwords": True}
        self.tm = text_model.TextModel(outcomes, texts, 'bag-of-words', options)

        # data_pipe = text_pipe.TextPipe()
        data = DataFrame({"y": outcomes, "text": texts})

        # Storing whether the outcome is a dummy:
        if set(data.y) == set([0, 1]):
            self.is_dummy_outcome = True

        N = data.shape[0]
        self.number_of_observations = N

        data.index = [str(x) for x in range(N)]

        parameters = self.tm.parameters

        # I trust a regression only if the optimal regularization parameter alpha
        # is strictly inside the grid.
        data['y_hat'] = self.tm.pipe.predict(texts)

        ridge = self.tm.pipe.named_steps['ridge_model']

        self.coef = ridge.coef_
        self.number_of_features = len(self.coef)
        self.std_X = ridge.std_X
        self.pipe = self.tm.pipe
        self.parameters = parameters
        self.parameters_display = parameters_display
        self.mean_outcome_in_groups = mean_outcome_in_groups(data.y, data.y_hat)
        self.percent_correct = share_correct(data.y, data.y_hat, verbose=verbose)
        self.outcome_summary = get_summary(outcomes)

        features = self.tm.pipe.named_steps['featurizer'].get_feature_names()

        self.features = [f.split("__")[1] for f in features]
        self.regression_table = self.get_regression_table()


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

    def adjust_probability(self, probability, text, known_phrases):
        # For known phrases, shift probability into known direction by similarity

        for (phrase, outcome) in known_phrases:
            sim_1, sim_2 = get_similarity(text, phrase)
            if sim_1 > .5 and sim_2 > .1:
                probability += (sim_1 + sim_2) / 2 * (outcome * 2 - 1)
        return probability

    def predict(self, texts):
        return self.tm.predict(texts)

    def get_features(self, text):
        """This takes a text and translates it into features. This function thus describes central logic of the TextModel"""
        # print self.tm.pipe
        x = Pipeline(self.tm.pipe.steps[0:-1]).transform([text]).toarray()[0]
        print(x)
        print("X")
        features = Series(self.features)[Series(x) >= 1]
        tab = self.regression_table
        coefficients_for_features = tab[tab.index.isin(features)]
        return coefficients_for_features

    def prediction_summary(self, text):
        summary = {}
        try:
            text = text.encode('utf-8', 'ignore')
        except UnicodeEncodeError:
            pass
        summary["predicted_value"] = self.predict(text)
        tab = self.regression_table
        tab_found = self.get_features(text)
        features_plus_minus = get_features_plus_minus(tab_found)
        summary["important_features_good_and_bad"] = features_plus_minus
        return summary


def get_text_outcome(content, outcome_variable, text_variable):
    data_file_new = open(content["file_name"], "r")
    text = data_file_new.read()
    data_file_new = open(content["file_name"], "r")
    data = pd.read_csv(data_file_new)
    data[text_variable].fillna("", inplace=True)
    texts = list(data[text_variable])
    # Converting into lower_case
    outcomes = list(data[outcome_variable])

    # Checking that these have the right type:
    if not all(isinstance(text, str) for text in texts):
        for i in range(len(texts)):
            if not isinstance(texts[i], str):
                pass
        raise TypeError("Texts are not all strings")

    return (texts, outcomes)


def get_content(data_file_string):
    structure = {}

    # Read the file using pandas. If it doesn't work, code the problem. If it does work
    # code the potential columns (string and binary)
    content = {}
    content["has_data"] = False
    if data_file_string:
        data_write = open("temp.txt", "w")
        data_write.write(data_file_string)
        data_write.close()
        data_file_new = open("temp.txt", "r")
        try:

            data = pd.read_csv(data_file_new)

            text_variables = []
            outcome_variables = []
            for (variable, typ) in data.dtypes.iteritems():
                if re.match("^[A-Za-z_-]*$", variable):
                    if typ in ["int64", "float64"]:
                        outcome_variables.append(variable)
                    if typ in ["object"]:
                        series = data[variable]
                        for obj in series:
                            if not isinstance(obj, basestring):
                                break
                        text_variables.append(variable)

            data_file_new = open("temp.txt", "r")

            # content["content"]=data_file_string
            content["text_variables"] = text_variables
            content["outcome_variables"] = outcome_variables

            if len(text_variables) > 0 and len(outcome_variables) > 0:
                content["data"] = data
                content["has_data"] = True
            else:
                content["error"] = "Could not find both text and numeric variable"

        except:
            content["error"] = "Could not read tabular structure of file"
        content["file_name"] = "temp.txt"
        content["original_file_name"] = "Submitted"

        return content

