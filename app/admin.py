# -*- encoding: utf-8 -*-
"""
License: MIT
Copyright (c) 2019 - present AppSeed.us
"""

from django.contrib import admin
from app.models import InsuranceClaim, LimeReport


@admin.register(InsuranceClaim)
class InsuranceClaimAdmin(admin.ModelAdmin):
    list_display = ("ID", "자동지급", "심사", "조사")


@admin.register(LimeReport)
class LimeReportAdmin(admin.ModelAdmin):
    list_display = [field.name for field in LimeReport._meta.get_fields()]
