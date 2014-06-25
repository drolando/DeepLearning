import sys, os
from os import listdir
from random import shuffle

if len(sys.argv) != 3:
    print "Use: python %s num_of_train_images num_of_test_images\n"%(sys.argv[0])
    sys.exit(1)

if not os.path.isdir('./ann'):
    print "Error: folder ann not found. Download the trecvid dataset.\n"
    sys.exit(1)
if not os.path.isdir('./train'):
    print "Error: folder train not found. Download the trecvid dataset.\n"
    sys.exit(1)
if not os.path.isdir('./test'):
    print "Error: folder test not found. Download the trecvid dataset.\n"
    sys.exit(1)

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
