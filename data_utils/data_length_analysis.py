from collections import defaultdict

f = open("../data/roi_event_dt=2018-04-25/path.txt")
d = defaultdict(int)
while True:
    line = f.readline()
    if not line:
        break

    length = len(line.split("\t"))

    d[length] += 1

d_tuple = sorted(d.items())

total = sum(d.values())


for ele in d_tuple:
    print(ele[0], ele[1], float(ele[1]/total))
