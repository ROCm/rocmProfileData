from django.db import models

# Create your models here.

class Version(models.Model):
    version = models.CharField(max_length=64, default='0.99')

class String(models.Model):
    string = models.CharField()

class KernelOp(models.Model):
    gridX = models.IntegerField(default=0)
    gridY = models.IntegerField(default=0)
    gridz = models.IntegerField(default=0)
    workgroupX = models.IntegerField(default=0)
    workgroupY = models.IntegerField(default=0)
    workgroupZ = models.IntegerField(default=0)
    lds = models.IntegerField(default=0)
    scratch = models.IntegerField(default=0)
    codeObject = model.ForeignKey(KernelCodeObject, on_delete=models.CASCADE)
    kernelName = model.ForeignKey(String, on_delete=models.CASCADE)
    kernelArgAddress = #64 bit int
    aquireFence = (none, agent, system)
    releaseFence = (...)


class KernelCodeObject()
    vgpr = models.IntegerField(default=0)
    sgpr = models.IntegerField(default=0)
    fbar = models.IntegerField(default=0)

class Api(models.Model):
    pid = models.IntegerField(default=0)
    tid = models.IntegerField(default=0)
    apiName = model.ForeignKey(String, on_delete=models.CASCADE)
    args = models.CharField()
    start = models.IntegerField(default=0)
    end = models.IntegerField(default=0)

class Ops(models.Model):
    gpuId = models.IntegerField(default=0)
    queueId = models.IntegerField(default=0)
    sequenceId = models.IntegerField(default=0)
    opType = model.ForeignKey(OpType, on_delete=models.CASCADE) 
    description = model.ForeignKey(String, on_delete=models.CASCADE)
    completionSignal = models.CharField(max_length=64)  #64 bit int
    start = models.IntegerField(default=0)
    end = models.IntegerField(default=0)

class CopyOp(models.Model):
    op = model.ForeignKey(Ops, on_delete=models.CASCADE)
    size = model.IntegerField(default=0)
    src
    dst
    sync
    pinned
    inputSignals = 

class BarrierOp()
    signalCount
    inputSignals = Foriegn (Signal,  opid->opid
    aquireFence = (none, agent, system)
    releaseFence = (...)


class OpType(models.Model):
    opName = models.CharField(max_length=64)


