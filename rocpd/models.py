from django.db import models



class Version(models.Model):
    version = models.CharField(max_length=64, default='0.99')
    #class Meta:
    #    db_table = 'version'

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

class KernelOp(Op):
    #op = models.OneToOneField(Ops, on_delete=models.PROTECT, primary_key=True)
    gridX = models.IntegerField(default=0)
    gridY = models.IntegerField(default=0)
    gridz = models.IntegerField(default=0)
    workgroupX = models.IntegerField(default=0)
    workgroupY = models.IntegerField(default=0)
    workgroupZ = models.IntegerField(default=0)
    groupSegementSize = models.IntegerField(default=0)
    privateSegementSize = models.IntegerField(default=0)
    codeObject = models.ForeignKey(KernelCodeObject, on_delete=models.PROTECT)
    kernelName = models.ForeignKey(String, on_delete=models.PROTECT)
    kernelArgAddress = models.CharField(max_length=18)  #64 bit int
    aquireFence = models.CharField(max_length=8)   #(none, agent, system)
    releaseFence = models.CharField(max_length=8)  #(none, agent, system)

class CopyOp(Op):
    #op = models.OneToOneField(Ops, on_delete=models.PROTECT, primary_key=True)
    size = models.IntegerField(default=0)
    src = models.IntegerField(default=0) # GPU id or -1
    dst = models.IntegerField(default=0) # GPU id or -1
    sync = models.BooleanField()
    pinned = models.BooleanField()

class BarrierOp(Op):
    #op = models.OneToOneField(Ops, on_delete=models.PROTECT, primary_key=True)
    signalCount = models.IntegerField()
    aquireFence = models.CharField(max_length=8)   #(none, agent, system)
    releaseFence = models.CharField(max_length=8)  #(none, agent, system)

#class InputSignal(models.Model)
#    op = models.ForeignKey(Ops, on_delete=models.PROTECT)
#    inputOp = models.ForeignKey(Ops, on_delete=models.PROTECT)

#class ApiOps(models.Model)
#    api = models.ForeignKey(Api, on_delete=models.PROTECT)
#    op = models.ForeignKey(Ops, on_delete=models.PROTECT)
