from django.contrib import admin
from textpredictions.models import PredictionModel, TextEntry

admin.site.register(PredictionModel)
admin.site.register(TextEntry)