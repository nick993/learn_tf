import sys
import os
import re

parentLocation = sys.argv[1]

for fileName in os.listdir(parentLocation):
    fileChars = re.split('(\(.*\))', fileName)
    newFileName = fileChars[1] + fileChars[0].rstrip() + fileChars[2]
    fileAbsLoc = parentLocation + '\\' + fileName
    newFileAbsLocation = parentLocation + '\\' + newFileName
    try:
        os.rename(fileAbsLoc, newFileAbsLocation)
        print('Successfully renamed file', fileAbsLoc, 'to', newFileAbsLocation)
    except IOError as e:
        print('Could not rename file', fileAbsLoc)
        print(e)
    print('')


