# -*- encoding: utf-8 -*-
"""
License: MIT
Copyright (c) 2019 - present AppSeed.us
"""
from django.db import models


class InsuranceClaim(models.Model):
    ID = models.IntegerField(primary_key=True)
    nur_hosp_yn = models.IntegerField()
    ac_ctr_diff = models.IntegerField()
    hsp_avg_optt_bilg_isamt_s = models.FloatField()
    hsp_avg_surop_bilg_isamt_s = models.FloatField()
    ar_rclss_cd = models.IntegerField()
    fds_cust_yn = models.IntegerField()
    hspz_dys_s = models.FloatField()
    inamt_nvcd = models.IntegerField()
    hsp_avg_diag_bilg_isamt_s = models.FloatField()
    blrs_cd = models.IntegerField()
    dsas_ltwt_gcd = models.IntegerField()
    dsas_avg_diag_bilg_isamt_s = models.FloatField()
    dsas_acd_rst_dcd = models.IntegerField()
    base_ym = models.IntegerField()
    kcd_gcd = models.IntegerField()
    hsp_avg_hspz_bilg_isamt_s = models.FloatField()
    optt_blcnt_s = models.FloatField()
    mtad_cntr_yn = models.IntegerField()
    heltp_pf_ntyn = models.IntegerField()
    prm_nvcd = models.IntegerField()
    surop_blcnt_s = models.FloatField()
    mdct_inu_rclss_dcd = models.IntegerField()
    dsas_avg_optt_bilg_isamt_s = models.FloatField()
    isrd_age_dcd = models.IntegerField()
    hspz_blcnt_s = models.FloatField()
    dsas_avg_surop_bilg_isamt_s = models.FloatField()
    urlb_fc_yn = models.IntegerField()
    dsas_avg_hspz_bilg_isamt_s = models.FloatField()
    smrtg_5y_passed_yn = models.IntegerField()
    ac_rst_diff = models.IntegerField()
    bilg_isamt_s = models.FloatField()
    optt_nbtm_s = models.FloatField()
    자동지급 = models.FloatField(null=True)
    심사 = models.FloatField(null=True)
    조사 = models.FloatField(null=True)
    target = models.CharField(max_length=15, null=True)

    class Meta:
        ordering = ("-ID",)


class LimeReport(models.Model):
    class Meta:
        unique_together = (('claim', 'rank'),)
    claim = models.ForeignKey(InsuranceClaim, on_delete=models.CASCADE, related_name="claim")  # N:1 FK
    rank = models.IntegerField()
    feature = models.CharField(max_length=50)
    local_exp = models.FloatField()
    discretized = models.TextField(null=True, max_length=100)
    value = models.FloatField()
