#!/usr/bin/env python3

import os
import sys
import xml.etree.ElementTree as ET
import pandas as pd
import glob
import csv

for e in glob.glob(sys.argv[1]+'*_*_*[0-9].xml'):
    print(e)
    tiles=[]
    prefix = os.path.basename(e).split('.')[0]
    cam_file = str(os.path.dirname(e)+'/'+prefix+'_Cameron.xml')
    tiles.append(e)
    tiles.append(cam_file)
    for fname in tiles:
        ##print(fname)
        root = ET.parse(fname).getroot()

        if fname != cam_file:
            marker_num_1=0
            inc=0
            for x in root[1][1]:
                if x.tag == "Marker":
                    marker_num_1 += 1
                else:
                    inc+=1

            marker_num_2=0
            inc=0
            for x in root[1][2]:
                if x.tag == "Marker":
                    marker_num_2 += 1
                else:
                    inc+=1
        else:
            marker_num_1=0
            inc=0
            for x in root[1][1]:
                if x.tag == "Marker":
                    marker_num_1 += 1
                else:
                    inc+=1

            marker_num_2=0
            inc=0
            for x in root[1][2]:
                if x.tag == "Marker":
                    marker_num_2 += 1
                else:
                    inc+=1

        #create df from xml
        if fname != cam_file:
            df_1_x = pd.concat([pd.DataFrame([root[1][1][i][0].text], columns=['x']) for i in range(inc,marker_num_1+inc)], ignore_index=True)
            df_1_y = pd.concat([pd.DataFrame([root[1][1][i][1].text], columns=['y']) for i in range(inc,marker_num_1+inc)], ignore_index=True)

            df_2_x = pd.concat([pd.DataFrame([root[1][2][i][0].text], columns=['x']) for i in range(inc,marker_num_2+inc)], ignore_index=True)
            df_2_y = pd.concat([pd.DataFrame([root[1][2][i][1].text], columns=['y']) for i in range(inc,marker_num_2+inc)], ignore_index=True)
            #merge x and y into one df
            green_coord_orig = pd.concat([df_1_x, df_1_y], axis=1)
            red_coord_orig = pd.concat([df_2_x, df_2_y], axis=1)
        else:
            df_1_x = pd.concat([pd.DataFrame([root[1][1][i][0].text], columns=['x']) for i in range(inc,marker_num_1+inc)], ignore_index=True)
            df_1_y = pd.concat([pd.DataFrame([root[1][1][i][1].text], columns=['y']) for i in range(inc,marker_num_1+inc)], ignore_index=True)

            df_2_x = pd.concat([pd.DataFrame([root[1][2][i][0].text], columns=['x']) for i in range(inc,marker_num_2+inc)], ignore_index=True)
            df_2_y = pd.concat([pd.DataFrame([root[1][2][i][1].text], columns=['y']) for i in range(inc,marker_num_2+inc)], ignore_index=True)
            #merge x and y into one df
            green_coord_edit = pd.concat([df_1_x, df_1_y], axis=1)
            red_coord_edit = pd.concat([df_2_x, df_2_y], axis=1)

        #save to csv
        if fname != cam_file:
            #green_coord_orig.to_csv(os.path.dirname(e)+'/'+(root[0][0].text).split('.')[0]+'_green.csv', index=False)
            #red_coord_orig.to_csv(os.path.dirname(e)+'/'+(root[0][0].text).split('.')[0]+'_red.csv', index=False)
            orig_green_tuples = [tuple(x) for x in green_coord_orig.to_numpy()]
            orig_red_tuples = [tuple(x) for x in red_coord_orig.to_numpy()]

        else:
            green_coord_edit.to_csv(os.path.dirname(e)+'/'+(root[0][0].text).split('.')[0]+'_Cameron_green.csv', index=False)
            red_coord_edit.to_csv(os.path.dirname(e)+'/'+(root[0][0].text).split('.')[0]+'_Cameron_red.csv', index=False)
            edit_green_tuples = [tuple(x) for x in green_coord_edit.to_numpy()]
            edit_red_tuples = [tuple(x) for x in red_coord_edit.to_numpy()]

    added_green, added_red = 0,0
    removed_green, removed_red = 0,0

    for i in edit_green_tuples:
        if i not in orig_green_tuples:
            added_green += 1
    for i in edit_red_tuples:
        if i not in orig_red_tuples:
            added_red += 1

    for i in orig_green_tuples:
        if i not in edit_green_tuples:
            removed_green += 1
    for i in orig_red_tuples:
        if i not in edit_red_tuples:
            removed_red += 1

    orig_stdout = sys.stdout
    f = open(os.path.dirname(e)+'/'+prefix+'_summary.txt', 'w')
    sys.stdout = f
    print(prefix+'.xml')
    print('')

    print('In the Green Channel:')
    PPV = round((green_coord_orig.shape[0]-removed_green)/(green_coord_orig.shape[0]-removed_green+removed_green)*100,2)
    TPR = round((green_coord_orig.shape[0]-removed_green)/(green_coord_orig.shape[0]-removed_green+added_green)*100,2)
    print("{} cells marked by machine".format(green_coord_orig.shape[0]))
    print("{} cells were added and {} cells were removed by Cameron".format(added_green, removed_green))
    print("Precision (PPV) = TP/(TP+FP) = {}%".format(PPV))
    print("Recall (TPR) = TP/(TP+FN) = {}%".format(TPR))
    print('')
    print('In the Red Channel:')
    PPV = round((red_coord_orig.shape[0]-removed_red)/(red_coord_orig.shape[0]-removed_red+removed_red)*100,2)
    TPR = round((red_coord_orig.shape[0]-removed_red)/(red_coord_orig.shape[0]-removed_red+added_red)*100,2)
    print("{} cells marked by machine".format(red_coord_orig.shape[0]))
    print("{} cells were added and {} cells were removed by Cameron".format(added_red, removed_red))
    print("Precision (PPV) = TP/(TP+FP) = {}%".format(PPV))
    print("Recall (TPR) = TP/(TP+FN) = {}%".format(TPR))
    sys.stdout = orig_stdout
    f.close()
    print('Done')
