import re
from matplotlib import pyplot as plt
width = []
height = []
size = []
with open('log.txt') as f:
    while True:
        str = f.readline()
        if not str:
            break
        if('tif image has shape') in str:
            if(len(re.findall(r"\d+",str))!=2):
                continue
            width.append(re.findall(r"\d+",str)[0])
            height.append(re.findall(r"\d+",str)[1])
print(len(height))
print(len(list((set(height)))))

for i in range(len(width)):
    if [width[i],height[i]] not in size:
        size.append([width[i],height[i]])
print(size)

plt.hist()