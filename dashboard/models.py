from django.db import models


class ReservoirMetaData(models.Model):
    stn_id = models.CharField(max_length=7, primary_key=True)
    name = models.CharField(max_length=255)
    lat = models.FloatField()
    lon = models.FloatField()
    fp = models.FloatField()
    dp = models.FloatField()
    info_flag = models.IntegerField()
    state = models.CharField(max_length=2)
    pf = models.FloatField()

    # slug = models.SlugField(max_length=255, unique=True, blank=True)  # Add slug field

    class Meta:
        db_table = 'tx_res_meta'

    def __str__(self):
        return self.name


class ReservoirDailyData(models.Model):
    date = models.DateField()
    cs = models.FloatField()
    precip = models.FloatField()
    avg_temp = models.FloatField()
    reservoir_name = models.CharField(max_length=255)
    # cs_id = models.CharField(max_length=8)
    cs_id = models.CharField(max_length=8, primary_key=True)  # Store cs_id as a CharField

    class Meta:
        db_table = 'reservoir_daily_data'
        # unique_together = ('date', 'reservoir_name')

    def clean_cs_id(self):
        """Remove leading zeros to match stn_id format."""
        return self.cs_id.lstrip('0')

    def __str__(self):
        return f"{self.reservoir_name}"
