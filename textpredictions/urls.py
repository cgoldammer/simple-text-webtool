from django.conf.urls import patterns, url, include
from django.contrib.staticfiles.urls import staticfiles_urlpatterns
# Uncomment the next two lines to enable the admin:
from django.contrib import admin

admin.autodiscover()
from textpredictions import views
from dajaxice.core import dajaxice_autodiscover, dajaxice_config

dajaxice_autodiscover()

urlpatterns = patterns('textpredictions.views',
                       url(dajaxice_config.dajaxice_url, include('dajaxice.urls')),
                       url(r'^admin/', include(admin.site.urls)),
                       url(r'^enter_text/model(?P<model_pk>\w+)$', 'enter_text', {'text_entry_pk': 0}),
                       url(r'^enter_text/model(?P<model_pk>\w+)/text(?P<text_entry_pk>\w+)$', 'enter_text'),
                       url(r'^technical$', views.TechnicalModelView.as_view(), name='models_technical'),
                       url(r'^models$', views.ModelView.as_view(), name='models'),
                       url(r'^model/(?P<model_pk>\w+)$', 'model', name='model'),
                       url(r'^about$', 'about', name='about'),
                       url(r'^$', views.ModelView.as_view(), ),
)
urlpatterns += staticfiles_urlpatterns()