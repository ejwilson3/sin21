import simple
import conf
import math

filename = "../data/auto93.csv"
sample = simple.Sample(conf.CONFIG)
sample.read(filename)
sample.sort()
branches = []
branch = []
fft = simple.Fft(sample, branch, branches)
n = 0
for branch in branches:
    print("tree " + str(n))
    n += 1
    el = "     "
    for leaf in branch:
        print(str(leaf.type) + el + leaf.txt + " " + str(leaf.then) + " (" + str(leaf.n) + ")")
        # print(leaf)
        el = " else"
    print("")
