import re
import time

firstline = []
types = []
datalist = []

runtime = time.time()
file = open("../data/thefile.csv")
data = file.readlines()
templine = ""
while(firstline == []):
    temp = data.pop(0).split("#")[0]
    temp = temp.strip(" \n")
    if len(temp) == 0:
        continue
    templine += temp
    if templine[-1] == ",":
        continue
    firstline = templine.split(",")
    for item in firstline:
        ch = re.search("\w", item)
        if ch.group().isupper():
            types.append('n')
        else:
            types.append('c')

num_cols = len(firstline)
dangling = ""
for line in data:
    temp = line.split("#")[0]
    temp = temp.strip()
    if len(temp) == 0:
        continue
    if temp[-1] == ",":
        dangling += temp
        continue
    temp = dangling + temp
    dangling = ""
    linelist = temp.split(",")
    for i in range(len(linelist)):
        linelist[i] = linelist[i].strip(" \n")
    linelist = [item for item in linelist if item != '']
    if (len(linelist) != num_cols):
        print("Length mismatch in line " + str(temp) + "; skipping line")
        continue
    for i in range(len(linelist)):
        if types[i] == 'n':
            try:
                linelist[i] = int(linelist[i])
            except:
                try:
                    linelist[i] = float(linelist[i])
                except:
                    if linelist[i] == 'TRUE':
                        linelist[i] = 1
                    elif linelist[i] == 'FALSE':
                        linelist[i] = 0
                    else:
                        print("NaN in number column in line " + str(temp) + "; skipping line")
                        continue
    datalist.append(linelist)
print("Ran in " + str(time.time() - runtime) + "s")
