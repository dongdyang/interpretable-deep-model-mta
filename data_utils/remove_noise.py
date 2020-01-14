import os
from collections import defaultdict


date = "25"
root_dir = "./roi_event_dt=2018-04-"+date
fileList = os.listdir(root_dir)
file_final = "./f_roi_event_dt=2018-04-"+date

file2 = open(file_final, "w")
context_dict = defaultdict(int)
for file_path in fileList:
    data_file_path = os.path.join(root_dir, file_path)
    file = open(data_file_path)
    lineNum = 0
    while True:
        line = file.readline()
        lineNum += 1
        lines = line.split("\t")
        if not line:
            break
        if lines[0]==r"\N" or lines[3] == r"\N":
            #print(file_path+"\t"+str(lineNum)+":\t"+line)
            continue

        context = lines[0] + "\t" + lines[7] + "\t" + lines[8] + "\t" +lines[9] + "\t" +lines[10] + "\t" + \
                lines[11] + "\t" + lines[12] + "\t" + lines[15] + "\t" + lines[17] + "\t" + \
                lines[19] + "\t" + lines[22] + "\t" +lines[23] + "\t" +lines[25] + "\t" +lines[26] + "\t" +lines[35] + "\n"
                
        if context in context_dict:
            continue

        file2.write(context)
        context_dict[context] = 1

    print("ONE FILE DONE")
    file.close()

file2.close()
print("------ALL DONE")

