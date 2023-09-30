
################################################################################
# Copyright (c) 2021 - 2023 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
################################################################################

from django.db import models


class Metadata(models.Model):
    tag = models.CharField(max_length=4096)
    value = models.CharField(max_length=4096)

class String(models.Model):
    string = models.CharField(max_length=4096)
    class Meta:
        indexes = [
             models.Index(fields=['string'])
        ]

class KernelCodeObject(models.Model):
    vgpr = models.IntegerField(default=0)
    sgpr = models.IntegerField(default=0)
    fbar = models.IntegerField(default=0)

class Api(models.Model):
    pid = models.IntegerField(default=0)
    tid = models.IntegerField(default=0)
    apiName = models.ForeignKey(String, related_name='+', on_delete=models.PROTECT)
    args = models.ForeignKey(String, related_name='+', on_delete=models.PROTECT)
    #ops = models.ManyToManyField(Op, through = 'ApiOps')
    ops = models.ManyToManyField('Op')
    start = models.IntegerField(default=0)
    end = models.IntegerField(default=0)

class Op(models.Model):
    gpuId = models.IntegerField(default=0)
    queueId = models.IntegerField(default=0)
    sequenceId = models.IntegerField(default=0)
    opType = models.ForeignKey(String, related_name='+', on_delete=models.PROTECT) 
    description = models.ForeignKey(String, related_name='+', on_delete=models.PROTECT)
    #inputSignals = models.ManyToManyField(Op, through = 'InputSignal')
    inputSignals = models.ManyToManyField('self')
    completionSignal = models.CharField(max_length=18)  #64 bit int
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
    codeObject = models.ForeignKey(KernelCodeObject, on_delete=models.PROTECT)
    kernelName = models.ForeignKey(String, on_delete=models.PROTECT)
    kernelArgAddress = models.CharField(max_length=18)  #64 bit int
    aquireFence = models.CharField(max_length=8)   #(none, agent, system)
    releaseFence = models.CharField(max_length=8)  #(none, agent, system)

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

class BarrierOp(Op):
    #op = models.OneToOneField(Ops, on_delete=models.PROTECT, primary_key=True)
    signalCount = models.IntegerField()
    aquireFence = models.CharField(max_length=8)   #(none, agent, system)
    releaseFence = models.CharField(max_length=8)  #(none, agent, system)

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

#class InputSignal(models.Model)
#    op = models.ForeignKey(Ops, on_delete=models.PROTECT)
#    inputOp = models.ForeignKey(Ops, on_delete=models.PROTECT)

#class ApiOps(models.Model)
#    api = models.ForeignKey(Api, on_delete=models.PROTECT)
#    op = models.ForeignKey(Ops, on_delete=models.PROTECT)
