from django.contrib import admin

# Register your models here.
from AICoach.models import ModelSpecification


class ModelingAdmin(admin.ModelAdmin):
    pass

admin.site.register(ModelSpecification, ModelingAdmin)