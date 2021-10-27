import simple
import conf
import math
import time

filename = "../data/auto93.csv"
# print(conf.CONFIG)
sample = simple.Sample(conf.CONFIG)
# read = sample.conf('forest')
# sample.conf['forest'] = (read[0], False, read[2], read[3], read[4], read[5])
sample.read(filename)
sample.sort()
key = ""
cols = []
ynames = []
for key in sample._conf:
    # all xs are nums
    if sample._conf[key][5]:
        cols.append(key.capitalize())
for y in sample._ys:
    cols.append(sample._col_info[y][1])
    ynames.append(sample._col_info[y][1])
cols.append("N+")

final_samples = []
final_bests = []
ranges = [30, 60, 125, 250, 500, 1000]
runtimes = [time.time()]
print("   " + str(cols))
for r in ranges:
    new_sample = simple.Sample(conf.CONFIG)
    # print(cols)
    new_sample.add(cols)
    # print("ADDED")
    for i in range(r):
        row = []
        # key = sample.shuffle_param()
        sample.shuffle_params()
        # read = sample._conf['samples']
        # sample._conf['samples'] = (read[0], 307, read[2], read[3], read[4], read[5])
        branch = []
        branches = []
        try:
            fft = simple.Fft(sample, branch, branches)
            for key in sample._conf:
                if sample._conf[key][5]:
                    row.append(sample._conf[key][1])
            tinysample = simple.Sample(conf.CONFIG)
            tinysample.add(ynames + ["N+"])
            for branch in branches:
                for leaf in branch:
                    tinysample.add(leaf.then + [leaf.n])
            tinysample.sort()
            for it in tinysample._rows[0]:
                row.append(it) 
            new_sample.add(row)
        except:
            # print(key)
            sample.spit_params()
            print(" ")
    final_samples.append(new_sample)
    new_sample.sort()
    new_branches = []
    branch = []
    fft = simple.Fft(new_sample, branch, new_branches)
    long_sample = simple.Sample(conf.CONFIG)
    long_sample.add(cols)
    for branch in new_branches:
        for leaf in branch:
            long_sample.add(leaf.dat)
    long_sample.sort()
    if len(long_sample._rows) == 0:
        print("Something went wrong. new_branches = ")
        print(new_branches)
        print("branches = ")
        print(branches)
        print("new_sample rows = ")
        print(new_sample._rows)
    print(str(r) + ": " + str(long_sample._rows[0]))
    final_bests.append(long_sample._rows[0])
    runtimes.append(time.time())

for t_idx, time in enumerate(runtimes[1:]):
    print(str(ranges[t_idx-1]) + " time: " + str(time - runtimes[t_idx-1]))
# for s_idx, samp in enumerate(final_samples):
    # samp.sort()
    # branches = []
    # branch = []
    # fft = simple.Fft(samp, branch, branches)
    # long_sample = simple.Sample(conf.CONFIG)
    # long_sample.add(cols)
    # for leaf in branches:
        # long_sample.add(leaf.dat)
    # long_sample.sort()
    # print(str(ranges[s_idx]) + str(long_sample._rows[0]))
    
