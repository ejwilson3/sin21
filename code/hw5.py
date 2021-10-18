import simple
import math

filename = "../data/auto93.csv"
sample = simple.Sample()
sample.read(filename)
sample.sort()
leaves, clusters = sample.leaf_clusters(True)
print("")
print("")

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
print("")
print("")

worst = sample.shoot([i._cells for i in leaves[0]])
best = sample.shoot([i._cells for i in leaves[-1]])
# for row in worst._rows:
    # print(row)
# for row in best._rows:
    # print(row)

for c_idx, col in enumerate(best._cols):
    if c_idx not in goalids:
        # print(col._vals)
        # print(c_idx)
        gen = col.discretize(worst._cols[c_idx])
        for it in gen:
            print(it)

print("")
print("")

for row in leaves[-1]:
    print("best " + str(row))
print("")
for row in leaves[0]:
    print("worst " + str(row))
# for good, bad in zip((
            # left_t = [i[1] for i in left]
