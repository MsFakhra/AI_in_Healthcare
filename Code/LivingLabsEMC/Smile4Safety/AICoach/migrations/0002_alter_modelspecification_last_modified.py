# Generated by Django 3.2.15 on 2022-10-17 23:18

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('AICoach', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='modelspecification',
            name='last_modified',
            field=models.DateTimeField(),
        ),
    ]
