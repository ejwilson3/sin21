import simple

files = ["../data/weather.csv", "../data/pom3a.csv", "../data/auto93.csv"]
sample1 = simple.Sample()
sample1.read(files[0])
sample2 = simple.Sample()
sample2.read(files[1])
sample3 = simple.Sample()
sample3.read(files[2])
sample3.sort()
print(sample3._col_info)
for i in range(5):
    print(sample3._rows[i])
print()
for i in range(5):
    print(sample3._rows[-1*i])
