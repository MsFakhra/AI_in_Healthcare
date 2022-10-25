from django.db import models

#superuser details: user = fjabeen, password = mani1234
# Create your models here.

class ModelSpecification(models.Model):
    model_id = models.AutoField(primary_key=True)
    model_name = models.TextField()
    model_specification = models.TextField()
    created_on = models.DateTimeField(auto_now_add=True)
    last_modified = models.DateTimeField()

#include deviation table