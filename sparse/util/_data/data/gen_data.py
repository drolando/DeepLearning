import sys
from os import listdir
from random import shuffle

adults = {}
file = open('./ann/Adult.ann', 'r')
for line in file:
    name = line.split(' ')[0].split('/')[-1]
    if line.split(' ')[-1].rstrip('\n') == 'N':
        adults[name] = 0
    elif line.split(' ')[-1].rstrip('\n') == 'P':
        adults[name] = 1
file.close()

out = open('train.txt', 'w')
files = listdir('train')
shuffle(files)
cnt = 0
for file in files:
    if file.split('.')[0] in adults:
        out.write('data/train/%s %d\n'%(file, adults[file.split('.')[0]]))
        cnt += 1
        if cnt >= int(sys.argv[1]):
            break
out.close()

out = open('val.txt', 'w')
files = listdir('test')
shuffle(files)
cnt = 0
for file in files:
    if file.split('.')[0] in adults:
        out.write('data/test/%s %d\n'%(file, adults[file.split('.')[0]]))
        cnt += 1
        if cnt >= int(sys.argv[2]):
            break
out.close()
