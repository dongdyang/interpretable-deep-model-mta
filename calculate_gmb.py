
import os
from collections import defaultdict
import numpy as np
import pickle

DATA_DATE = "2018-07-13"
root_dit = "./shap_value/shap_value_"
filenames = os.listdir(root_dit + DATA_DATE)


indexes = []
weight_all = defaultdict(float)
gmb_all = defaultdict(float)

gmb_map = defaultdict(float)
file = open("./shap_value/gmb_"+DATA_DATE+".txt")
for index, line in enumerate(file.readlines()):
    gmb_map[index] = float(line)
file.close()

for filename in filenames:
    file_path = os.path.join(root_dit + DATA_DATE, filename)
    file = open(file_path)
    try:
        for line in file.readlines():
            context = line.split("\t")
            index = int(context[0])
            indexes.append(index)

            channels = []
            shap_values = []
            for ele in context[1:]:
                temp = ele.replace("\n", "").split(",")
                channels.append(temp[0])
                shap_values.append(float(temp[1]))


            shap_values = [ele if ele > 0 else 0 for ele in shap_values]
            shap_values = np.array(shap_values)
            total = sum(shap_values)
            if total != 0:
                shap_values /= total

            for event_i in range(len(channels)):
                temp1 = channels[event_i]
                temp2 = shap_values[event_i]
                weight_all[temp1] += temp2
                gmb_all[temp1] += gmb_map[index] * temp2
    except:
        print("fail to read:"+file_path)


for channle in weight_all:
    weight_all[channle] /= len(indexes)


pickle.dump(indexes, open(DATA_DATE+'_indexes.pkl', 'wb'))


print(weight_all)
# dump information to that file
pickle.dump(weight_all, open(DATA_DATE+'_weight_all.pkl', 'wb'))

print(gmb_all)
pickle.dump(weight_all, open(DATA_DATE+'_gmb_all.pkl', 'wb'))


print("DONE")




