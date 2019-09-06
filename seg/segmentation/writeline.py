import sys 
sys.path.insert(0,'/home/fapsros/anaconda3/lib/python3.7/site-packages')
import random 

lines = []

'''
for i in range(3721):
    lines.append('{:04d}\n'.format(i))
'''

for i in range(2479, 3721):
    lines.append('{:04d}\n'.format(i))

random.shuffle(lines)

trainlines = []
testlines = []

for i in range(len(lines)-900, len(lines)):
    trainlines.append(lines[i])

for j in range(len(lines)-900):
    testlines.append(lines[j])

'''
print(len(trainlines))
print(len(testlines))
'''


fo = open('segstift_test.txt','w+')
fo.writelines(testlines)
fo.close()

fo = open('segstift_train.txt', 'w+')
fo.writelines(trainlines)
fo.close()