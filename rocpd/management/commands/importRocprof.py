from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from rocpd.models import Version, String, Op, Api
import re

class Command(BaseCommand):
    help = 'Import data from an RPT text file'

    def add_arguments(self, parser):
        parser.add_argument('--ops_input_file', type=str, help="hcc_ops_trace.txt from rocprofiler")
        parser.add_argument('--api_input_file', type=str, help="hip_api_trace.txt from rocprofiler")
        parser.add_argument('--preview', action='store_true', help='Preview import only')

    def handle(self, *args, **options):
      with transaction.atomic():
        apiKeys = {}
        if options['api_input_file']:
            print(f"Importing hip api calls from {options['api_input_file']}")

            exp = re.compile("^(\d*):(\d*)\s+(\d*):(\d*)\s+(\w+)\((.*)\).*$")
            infile = open(options['api_input_file'], 'r', encoding="utf-8")
            count = 0
            for line in infile:
                m = exp.match(line)
                if m:
                    entry = Api(pid = m.group(3) \
                                ,tid = m.group(4) \
                                ,start = m.group(1) \
                                ,end = m.group(2) \
                                )
                    apiName = None
                    apiArgs = None
                    try:
                        apiName = String.objects.get(string=m.group(5))
                    except:
                        apiName = String(string=m.group(5))
                        apiName.save()
                    try:
                        apiArgs = String.objects.get(string=m.group(6))
                    except:
                        apiArgs = String(string=m.group(6))
                        apiArgs.save()
                    entry.apiName = apiName
                    entry.args = apiArgs
                    entry.save()
                    apiKeys[count+1]=entry.id

                count = count + 1
                if count % 100 == 99:
                    self.stdout.write(f"{count+1}")
                #if count > 3000: break
            infile.close()

        if options['ops_input_file']:
            print(f"Importing hcc ops from {options['ops_input_file']}")

            exp = re.compile("^(\d*):(\d*)\s+(\d*):(\d*)\s+(\w+):(\d*).*$")
            infile = open(options['ops_input_file'], 'r', encoding="utf-8")
            count = 0
            for line in infile:
                m = exp.match(line)
                if m:
                    entry=Op(gpuId = m.group(3) \
                             ,queueId = m.group(4) \
                             ,start = m.group(1) \
                             ,end = m.group(2) \
                            )
                    opName = None
                    opDescription = None
                    try:
                        opName = String.objects.get(string=m.group(5))
                    except:
                        opName = String(string=m.group(5))
                        opName.save()
                    try:
                        opDescription = String.objects.get(string="")
                    except:
                        opDescription = String(string="")
                        opDescription.save()

                    entry.opType = opName
                    entry.description = opDescription
                    entry.save()

                    # Look up and link the related API call
                    try:
                        #print(f"{entry.opType}  {int(m.group(6))} -> {apiKeys[int(m.group(6))]}")
                        apiEntry = Api.objects.get(pk=apiKeys[int(m.group(6))])
                        apiEntry.ops.add(entry)
                        apiEntry.save()
                    except:
                        pass
                count = count + 1
                if count % 100 == 99:
                    self.stdout.write(f"{count+1}")
                #if count > 100: break
            infile.close()

