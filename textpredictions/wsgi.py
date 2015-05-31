import os, site, sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../'))

os.environ["DJANGO_SETTINGS_MODULE"] = "textpredictions.settings"  # see footnote [2]

from django.core.wsgi import get_wsgi_application

_application = get_wsgi_application()


def application(environ, start_response):
    return _application(environ, start_response)