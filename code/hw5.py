import simple
import math

filename = "../data/auto93.csv"
sample = simple.Sample()
sample.read(filename)
sample.sort()
leaves, clusters = sample.leaf_clusters(True)

# clusters = []
# for leaf in leaves:
    # clusters.append(leaf[math.floor(len(leaf)/2)])
# clusters.sort()
# print(*clusters, sep="\n")

# for row in clusters:
    # row._debug = True
# 
# print(clusters[0])
# print(clusters[1])
# print(clusters[0] < clusters[1])


goalids = []
goalnam = []
for c_idx, info in enumerate(sample._col_info):
    if info[0] == "y":
        goalids.append(c_idx)
        goalnam.append(info[1])
print(goalnam)
print("")

for cluster in clusters:
    goals = []
    # med = leaf[math.floor(len(leaf)/2)]
    for c_idx in goalids:
        goals.append(cluster[0]._cells[c_idx])
    print(goals)

worst = clusters[0]
best = clusters[-1]
goodxs = []
badxs = []
# for good, bad in zip((
            # left_t = [i[1] for i in left]
