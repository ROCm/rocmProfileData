from django.core.management.base import BaseCommand, CommandError
from rocpd.models import Op

class Command(BaseCommand):
    help = 'Import data from an RPT text file'

    def add_arguments(self, parser):
        parser.add_argument('input_file', nargs='+')
        parser.add_argument('--preview', action='store_true', help='Preview import only')

    def handle(self, *args, **options):
        infile = open(options['input_file'][0], 'r', encoding="utf-8")
        count = 0
        for line in infile:
            count = count + 1
            fields = line.split(';')
            print(fields)

        self.stdout.write(self.style.SUCCESS('Imported %s events' % count))
