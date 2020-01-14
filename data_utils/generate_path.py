

import os
from collections import defaultdict
import types
import re
import sys


date = "25"
clean_file = "../../roi_event_dt=2018-04-"+date

path_dict = {}

file = open(clean_file)
print("DONE0")

while True:
    line = file.readline()
    if not line:
        break

    context = line.split("\t")
    #lines[0] + "\t" + lines[7] + "\t" + lines[8] + "\t" + lines[9] + "\t" + lines[10] + "\t" + \
    #lines[11] + "\t" + lines[12] + "\t" + lines[15] + "\t" + lines[17] + "\t" + lines[19] + "\t" + \
    #lines[22] + "\t" + lines[23] + "\t" + lines[25] + "\t" + lines[26] + "\t" + lines[35] + "\n"

    channel_id = context[0]
    use_id = context[4]
    click_event_ts = context[9]
    click_event_rank = context[14]

    if use_id not in path_dict:
        path_dict[use_id] = {}
    if click_event_ts not in path_dict[use_id]:
        path_dict[use_id][click_event_ts] = [click_event_rank, channel_id]


file.close()
print("DONE1")

file_gmb_path1 = "../data/roi_event_dt=2018-04-"+date+"/path"
file1 = open(file_gmb_path1, 'w')

file_gmb_path2 = "../data/roi_event_dt=2018-04-"+date+"/time_path"
file2 = open(file_gmb_path2, 'w')

for ele in path_dict:
    temp = path_dict[ele]
    click_event_tss = sorted(path_dict[ele])
    path = ""
    time_path = ""
    #device_path = ""

    last_channel_id = ""
    for click_event_ts in click_event_tss:
        click_event_rank = temp[click_event_ts][0]
        channel_id = temp[click_event_ts][1]
        #if channel_id == "15":
        #    channel_id = "6"
        #    print(channel_id)

        #if last_channel_id != channel_id: #decide deduplicate or not
        path += channel_id + "\t"
        time_path += click_event_ts + "\t"
            #device_path += mobiledesktop(brower_id) + "\t"

        #last_channel_id = channel_id
        if int(click_event_rank) == 1:
            file1.write(path+"\n")
            file2.write(time_path+"\n")

            #file3.write(device_path+"\n")
            path = ""
            time_path = ""
            last_channel_id = ""

print("DONE2")
file1.close()
file2.close()
#file3.close()

print("DONE")



