
import matplotlib.pyplot as plt
import numpy as np

file = open("./shap_value/shap_value_2018-07-13/start_0_end_3125.txt")
lines = file.readlines()
number = 0

channel_names = {
'1':'Display',
'2':'Paid Search',
'3':'Natural Search',
'6':'Affiliate',
'9':'Shopping Comparison',
'13':'Social Media',
'14':'AdCommerce',
'15':'Partner Integration',
'16':'Others',
'17':'SEM',
'18':'Exclusion',
'19':'Unassigned',
'23':'Partner Integration - Organic',
'25':'Paid Search  Brand',
'26':'Natural Search - Gbase',
'27':'Shopping Comparison  - SDC',
'28':'Non-IM Mktg Initiatives',
'32':'Programmatic',
'33':'Paid Social'
}


for line in lines:
    context = line.split("\t")

    if len(context) < 5 or len(context)> 12:
        continue


    if number > 10:
        break
    number += 1

    importances = []
    channels = []
    for ele in context[1:]:
        temp = ele.split(",")
        channels.append(channel_names[temp[0]])
        #importances.append(float('%.2f' % float(temp[1])))
        importances.append(float(temp[1]))

    temp = [ele if ele > 0 else 0 for ele in importances]

    temp_sum = sum(temp)
    importances = np.array(importances)
    if temp_sum == 0:
        print("Error 0")
    importances /= temp_sum

    channels = np.array(channels)
    indices = range(len(channels))

    plt.figure()
    plt.title("Weight of each Click Event")


    n = len(channels)
    #plt.figure(figsize=(50, 50))

    plt.barh(bottom=range(n), height=0.25, width=importances[indices], color="g")
    # If you want to define your own labels,
    # change indices to a list of labels on the following line.
    #plt.xticks()
    plt.yticks(range(n), channels[indices], rotation=70)
    plt.xlim([0, 0.75])
    plt.ylim([-1, n])


    #plt.show()
    plt.savefig("./images/"+context[0]+'.png')
