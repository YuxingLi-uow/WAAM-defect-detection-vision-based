import sys
import os


# class Logger(object):
#
#     def __init__(self, filename="Default.log"):
#         self.terminal = sys.stdout
#         self.log = open(filename, "a")
#
#     def write(self, message):
#         self.terminal.write(message)
#         self.log.write(message)

sys.stdout = open('LogTrain.txt', 'a')

path = os.path.abspath(os.path.dirname(__file__))
type = sys.getfilesystemencoding()

# f = open('LogTrain.txt', 'a')

print(path)
print(os.path.dirname(__file__))
print('------------------')

sys.stdout.close()

# sys.stdout = Logger('LogTrain.txt')











