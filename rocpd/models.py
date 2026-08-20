
# Copyright (C) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from django.db import models


class Metadata(models.Model):
    tag = models.CharField(max_length=4096)
    value = models.CharField(max_length=4096)

# Commonly repeated strings - e.g. enums
class String(models.Model):
    string = models.CharField(max_length=4096)
    class Meta:
        indexes = [
             models.Index(fields=['string'])
        ]
# Unique strings - rarely/never repeating, not deduped
class UString(models.Model):
    string = models.CharField(max_length=4096)
    class Meta:
        indexes = [
             models.Index(fields=['string'])
        ]

class Api(models.Model):
    pid = models.IntegerField(default=0)
    tid = models.IntegerField(default=0)
    domain = models.ForeignKey(String, related_name='+', on_delete=models.PROTECT)
    category = models.ForeignKey(String, related_name='+', on_delete=models.PROTECT)
    apiName = models.ForeignKey(String, related_name='+', on_delete=models.PROTECT)
    args = models.ForeignKey(UString, related_name='+', on_delete=models.PROTECT)
    #ops = models.ManyToManyField(Op, through = 'ApiOps')
    ops = models.ManyToManyField('Op')
    start = models.IntegerField(default=0)
    end = models.IntegerField(default=0)

#class Annotation(models.Model):
#    domain = models.ForeignKey(String, related_name='+', on_delete=models.PROTECT)
#    category = models.ForeignKey(String, related_name='+', on_delete=models.PROTECT)
#    args = models.ForeignKey(UString, related_name='+', on_delete=models.PROTECT)

class Op(models.Model):
    gpuId = models.IntegerField(default=0)
    queueId = models.IntegerField(default=0)
    sequenceId = models.IntegerField(default=0)
    opType = models.ForeignKey(String, related_name='+', on_delete=models.PROTECT) 
    description = models.ForeignKey(String, related_name='+', on_delete=models.PROTECT)
    #inputSignals = models.ManyToManyField(Op, through = 'InputSignal')
    #inputSignals = models.ManyToManyField('self')
    #completionSignal = models.CharField(max_length=18)  #64 bit int
    start = models.IntegerField(default=0)
    end = models.IntegerField(default=0)

class KernelApi(Api):
    #api = models.OneToOneField(Api, on_delete=models.PROTECT, primary_key=True)
    stream = models.CharField(max_length=18)
    gridX = models.IntegerField(default=0)
    gridY = models.IntegerField(default=0)
    gridZ = models.IntegerField(default=0)
    workgroupX = models.IntegerField(default=0)
    workgroupY = models.IntegerField(default=0)
    workgroupZ = models.IntegerField(default=0)
    groupSegmentSize = models.IntegerField(default=0)
    privateSegmentSize = models.IntegerField(default=0)
    kernelName = models.ForeignKey(String, on_delete=models.PROTECT)
    #codeObject = models.ForeignKey(KernelCodeObject, on_delete=models.PROTECT)
    #kernelArgAddress = models.CharField(max_length=18)  #64 bit int
    #aquireFence = models.CharField(max_length=8)   #(none, agent, system)
    #releaseFence = models.CharField(max_length=8)  #(none, agent, system)

class CopyApi(Api):
    #api = models.OneToOneField(Api, on_delete=models.PROTECT, primary_key=True)
    stream = models.CharField(max_length=18)
    size = models.IntegerField(default=0)
    width = models.IntegerField(default=0)
    height = models.IntegerField(default=0)
    kind = models.IntegerField(default=0) # enum
    dst = models.CharField(max_length=18)
    src = models.CharField(max_length=18)
    dstDevice = models.IntegerField(default=0) # GPU id or -1
    srcDevice = models.IntegerField(default=0) # GPU id or -1
    sync = models.BooleanField()
    pinned = models.BooleanField()

class Counter(models.Model):
    op = models.ForeignKey(Op, related_name='+', on_delete=models.PROTECT)
    name = models.ForeignKey(String, related_name='+', on_delete=models.PROTECT)
    value = models.FloatField()

class Monitor(models.Model):
    class DeviceType(models.TextChoices):
        GPU = 'gpu'
        CPU = 'cpu'
    class MonitorType(models.TextChoices):
        MCLK = "mclk"
        SCLK = "sclk"
        TEMP = "temp"
        POWER = "power"
        FAN = "fan%"
        VRAM = "vram%"
        GPU = "gpu%"
    deviceType = models.CharField(max_length = 16, choices = DeviceType.choices)
    deviceId = models.IntegerField(default=0)
    monitorType = models.CharField(max_length = 16, choices = MonitorType.choices)
    start = models.IntegerField(default=0)
    end = models.IntegerField(default=0)
    value = models.CharField(max_length=255)

class StackFrame(models.Model):
    api_ptr = models.ForeignKey(Api, related_name='+', on_delete=models.PROTECT)
    depth = models.IntegerField(default=0)
    name = models.ForeignKey(String, related_name='+', on_delete=models.PROTECT)


#class ApiOps(models.Model)
#    api = models.ForeignKey(Api, on_delete=models.PROTECT)
#    op = models.ForeignKey(Ops, on_delete=models.PROTECT)
