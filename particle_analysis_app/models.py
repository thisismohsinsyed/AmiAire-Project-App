from django.db import models

class Result(models.Model):
    location_longitude = models.FloatField()
    location_latitude = models.FloatField()
    experiment_start_date = models.DateField()
    experiment_end_date = models.DateField()
    particle_count_results = models.JSONField()
    original_image = models.TextField()
    roi_extracted = models.TextField()