class FileBuffer:

   def __init__(self, file):
       self.file = open(file, 'w')

   def write(self, *argv):
       print(*argv, file=self.file)
       print(*argv)
