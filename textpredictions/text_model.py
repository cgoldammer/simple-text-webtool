"""This module contains all functions for simple-text-analysis. The package
allows you to build a predictive model from text in one line of code. 
This package takes care of a lot of non-trivial choices (such as text 
cleaning, estimation, and validation, via
sensible defaults.

Example
-------
The following shows that it's easy to use the module::

    from text_model_functions import TextModel
    # A predictive text model requires outcomes and texts
    texts=["hello", "yes", "no", "why", "is", "hello",
           "yes", "no", "why", "is", "I am John Lennon"]
    outcomes=range(len(texts_entities))

    # Building a text model takes one line
    text_model=TextModel(outcomes,texts,'bag-of-words')
    # A text model allows you to predict the outcome for an arbitrary text
    text_model.predict("Jack Lennon")

"""

import numpy as np
from pandas import DataFrame, Series
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Ridge
import re
import nltk
# This is required to make NLTK work with virtual environments.
# Change the environment before using.
nltk.data.path.append("/var/www/textprediction/nltk_data/")
import pickle
from sklearn.grid_search import GridSearchCV
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob

class RidgeWithStats(Ridge):
    def fit(self, X, y, sample_weight=1.0):
        self.std_X = DataFrame(X.toarray()).std()
        return Ridge.fit(self, X, y, sample_weight)


# Define a global dictionary with class subjects.
def module_from_name(module):
    if module == "bag-of-words":
        return ("bag-of-words", CountVectorizer())
    if module == "emotions":
        return ("emotions", EmotionFeaturizer())
    if module == "entities":
        return ("entities", NamedEntityFeaturizer())


def modules_to_dictionary(modules):
    """The modules argument can be provided in a wide variety of types 
    (string, list,dictionary). Internally, this will get translated to a 
    list (of module names and modules) and a dictionary of options."""
    modules_list = []
    options = {}
    # isinstance. And transform into dictionary. Also use 'for' regardless
    # of type
    if type(modules) == str:
        modules_list.append(module_from_name(modules))
    if type(modules) == list:
        for module in modules:
            modules_list.append(module_from_name(module))
    if type(modules) == dict:
        for module in modules.keys():
            modules_list.append(module_from_name(module))

    return modules_list, options


def named_entities(text, types=None):
    """This functions returns named entities from a text.
    Adapted from emh's code (http://stackoverflow.com/users/2673189/emh)

    Parameters
    ----------
    text: str
        UTF-8 string
    types: list of strings
        Currently the list can include only "PERSON" and "ORGANIZATION"

    Returns
    -------
    dict
        Dictionary with one entry for each type of entity. For each of these 
        entries, contains a list of strings with found entities
    """
    if not types:
        types = ["PERSON", "ORGANIZATION"]
    named_entities = {"PERSON": [], "ORGANIZATION": []}
    tokens = nltk.tokenize.word_tokenize(text)
    pos = nltk.pos_tag(tokens)
    sentt = nltk.ne_chunk(pos, binary=False)
    for type_ in types:
        for subtree in sentt.subtrees(filter=lambda t: t.label() == type_):
            entity = ""
            for leaf in subtree.leaves():
                entity = entity + " " + leaf[0]
            named_entities[type_].append(entity.strip())
    return named_entities


def entity_dict_to_list(entity_dict):
    entities = []
    # Note: Use iterators.
    for type_ in entity_dict.keys():
        ent_type = entity_dict[type_]
        entities.extend(["ENTITY__" + type_ + "_" + e for e in ent_type])
    return entities


def position_list(targets, sources, verbose=False):
    positions = [(target in sources) for target in targets]
    positions = 1 * Series(positions)
    return list(positions)


class BaseTransformer:
    def fit_transform(self, X, y, **fit_params):
        transformed = self.fit(X, y, **fit_params).transform(X, **fit_params)
        return transformed

    def fit(self, X, y, **fit_params):
        return self

    def transform(self, X, **fit_params):
        pass


# Start from base class and then override function transform_word
class Lemmatizer(BaseTransformer):
    """This is a transformer for lemmatization."""
    wnl = WordNetLemmatizer()

    def transform(self, X, **fit_params):
        X_lemmatized = [" ".join(
            [self.wnl.lemmatize(word) for word in TextBlob(text).words])
                        for text in X]
        return X_lemmatized

    def get_params(self, deep=False):
        return {}


class NamedEntityFeaturizer(BaseTransformer):
    """This is a transformer that turns text into named entities."""
    types = ["PERSON", "ORGANIZATION"]
    entities = []
    entities_set = None

    def fit(self, X, y, **fit_params):
        text_all = " ".join(X)
        entities = named_entities(text_all, self.types)
        self.entities = entity_dict_to_list(entities)
        self.entities_set = set(self.entities)
        return self

    def transform(self, X, **fit_params):
        X_data = []
        for text in X:
            entities = named_entities(text, self.types)
            entities_in_row = entity_dict_to_list(entities)
            positions = position_list(self.entities_set, entities_in_row)
            X_data.append(positions)
        X_data = np.array(X_data)
        if X_data.shape[1] == 0:
            raise ValueError("No named entities in training data!")
        return X_data

    def get_params(self, deep=False):
        return {}


class EmotionFeaturizer(BaseTransformer):
    """This class is used to extract macro-features of the text.
    Currently, it includes sentiment and subjectivity"""
    types = ["polarity", "subjectivity"]

    def value_given_type(self, type_, text):
        sentiment = TextBlob(text).sentiment._asdict()
        return sentiment[type_]

    def transform(self, X, **fit_params):
        X_data = []
        for text in X:
            text_data = []
            # Use list comprehension
            for type_ in self.types:
                text_data.append(self.value_given_type(type_, text))
            X_data.append(text_data)
        X_data = np.array(X_data)
        return X_data

    def get_params(self, deep=False):
        return {}


class TextModel:
    """This is the main class from this module. It allows you to build a
    text model given outcomes, texts, text modules used, and options."""
    pipe = None
    regression_table = None

    def __init__(self, outcomes, texts, modules, options=None, verbose=False):

        # Setting the default options
        if not options:
            options = {'lemmatize': False,
                       'lowercase': False,
                       'remove-stopwords': True}

        data = DataFrame({"y": outcomes, "text": texts})
        N = data.shape[0]
        data.index = [str(x) for x in range(N)]

        # Defining the alphas for the cross-validation. Note that
        # alpha scales proportionally with the number of observations.
        number_of_alphas = 5
        logspace_min = -2
        alphas = Series(np.logspace(logspace_min,
                                    logspace_min + number_of_alphas - 1,
                                    number_of_alphas)) * N

        text_cleaner = TextCleaner(**options)

        modules_list, _ = modules_to_dictionary(modules)
        if len(modules_list) == 0:
            raise ValueError("No modules specified or found.")

        feature_union = FeatureUnion(modules_list)
        ridge_model = RidgeWithStats()

        pipeline_list = [('cleaner', text_cleaner)]
        if options.get('lemmatize'):
            pipeline_list.append(('lemmatizer', Lemmatizer()))
        pipeline_list.append(('featurizer', feature_union))
        pipeline_list.append(('ridge_model', ridge_model))

        pipe = Pipeline(pipeline_list)

        parameters_initial = {'ridge_model__normalize': True}
        # If bag-of-words is included, add the relevant parameters
        if 'bag-of-words' in modules:
            def vec_name(param):
                return ('featurizer__bag-of-words__' + param)

            parameters_initial[vec_name('lowercase')] = False
            parameters_initial[vec_name('ngram_range')] = (1, 1)
            if options.get('remove-stopwords'):
                parameters_initial[vec_name('stop_words')] = "english"

        parameter_for_gridsearch = {'ridge_model__alpha': tuple(alphas)}

        pipe.set_params(**parameters_initial)

        # Pick meta-parameters using grid_search.
        grid_result = self.grid_search(data, pipe, parameter_for_gridsearch)
        (parameters_gridsearch_result, pipe) = grid_result
        # I trust a regression only if the optimal regularization
        # parameter alpha is strictly inside the grid.
        if parameters_gridsearch_result["ridge_model__alpha"] == max(alphas):
            raise ValueError("Regularization parameter hitting upper bound")

        # The full parameters consist of the initial values
        # and the parameters found through the grid-search
        parameters = parameters_initial
        for (key, value) in parameters_gridsearch_result.iteritems():
            parameters[key] = value

        data['y_hat'] = pipe.predict(texts)
        if verbose:
            print "Parameters from grid-search:"
            print parameters_gridsearch_result
            print "Final parameters:"
            print parameters

        # Keeping the regression, its coefficients, and the pipe as attributes
        ridge = pipe.named_steps["ridge_model"]
        self.coef = ridge.coef_
        self.pipe = pipe
        self.parameters = parameters


    def grid_search(self, train, pipe, params):
        """
        Train a model, optimizing over meta-parameters using sklearn's
        grid_search. Returns the best estimator from a sklearn.grid_search
        operation
        (See http://scikit-learn.org/stable/modules/grid_search.html)
        
        Function contributed by Ryan Wang.
        """
        grid_search = GridSearchCV(estimator=pipe, param_grid=params)
        grid_search.fit(train.text, train.y)
        return grid_search.best_params_, grid_search.best_estimator_

    def save_pipe(self, fp):
        pickle.dump(self, fp)

    def load_pipe(self, fp):
        m = pickle.load(fp)
        self.pipe = m.pipe

    def predict(self, texts):
        if isinstance(texts, basestring):
            texts = [texts]
        predictions = self.pipe.predict(texts)
        if (len(texts) == 1):
            return predictions[0]
        return predictions


class TextCleaner(BaseTransformer):
    """This function takes car of cleaning the text before it's featurized.

    Parameters
    ----------
    lowercase: bool
        Flag to convert the text to lowercase, defaults to false

    Returns
    -------
    class
        A transformer that can be used in a pipeline
    """

    lowercase = False
    options = {}

    def __init__(self, **kwargs):
        self.options = kwargs
        if kwargs is not None:
            if "lowercase" in self.options:
                self.lowercase = self.options["lowercase"]

    def transform(self, X, **fit_params):
        rx = re.compile("[^a-zA-Z]+")
        X_cleaned = [rx.sub(' ', str(t)).strip() for t in X]
        if self.lowercase:
            X_cleaned = [t.lower() for t in X_cleaned]
        return X_cleaned

    def get_params(self, deep=True):
        return self.options

    def set_params(self, **parameters):
        self.options = parameters
