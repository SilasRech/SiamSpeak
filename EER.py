import numpy as np
import sys
import numpy as np


def EER(clients, impostors):
    threshold = np.arange(-1,1,0.1) #[-0.5,-0.4,-0.3,-0.2,-0.1,0.,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
    FAR = []
    FRR = []
    for th in threshold:
        far = 0.0
        for score in impostors:
            if score.item() > float(th):
                far += 1
        frr = 0.0
        for score in clients:
            if score.item() <= float(th):
                frr += 1
        FAR.append(far/impostors.size)
        FRR.append(frr/clients.size)
    ERR = 0.
    dist = 1.
    for far, frr in zip(FAR,FRR):
        if abs(far-frr) < dist:
            ERR = (far+frr)/2
            dist = abs(far-frr)
    return float("{0:.3f}".format(100*ERR))

filename = sys.argv[1]

lst = open(filename,'r')

scores = []
lines = lst.readlines()
for x in lines:
     scores.append(np.float(x.split()[0]))

c = np.array(scores[0::2])
i = np.array(scores[1::2])

print('EER is : %.3f' %EER(c,i))
