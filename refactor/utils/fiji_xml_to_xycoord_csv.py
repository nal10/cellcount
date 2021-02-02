#!/usr/bin/env python3

#This script is for parsing Fiji xml annotation output and putting it in csv coordinate format
##Green annotations are type 1
##Red annotations are type 2

#Usage:
#python parse_xml.py <directory path>
#i.e. python fiji_xml_to_xycoord_csv.py /Users/elyse.morin/Desktop/xml_files/

import os
import sys
import xml.etree.ElementTree as ET
import pandas as pd
import glob


#Count number of cells annotated for each type



for i in os.listdir(sys.argv[1]):
    if i.endswith('.xml'):
        filename = glob.glob(sys.argv[1]+'/'+i)[0]
        print(filename)
        root = ET.parse(filename).getroot()

        marker_num_1=0
        for x in root[1][1]:
            if x.tag == "Marker":
                marker_num_1 += 1
            else:
                pass

        marker_num_2=0
        for x in root[1][2]:
            if x.tag == "Marker":
                marker_num_2 += 1
            else:
                pass

        #create df from xml
        df_1_x = pd.concat([pd.DataFrame([root[1][1][i][0].text], columns=['x']) for i in range(1,marker_num_1+1)], ignore_index=True)
        df_1_y = pd.concat([pd.DataFrame([root[1][1][i][1].text], columns=['y']) for i in range(1,marker_num_1+1)], ignore_index=True)

        df_2_x = pd.concat([pd.DataFrame([root[1][2][i][0].text], columns=['x']) for i in range(1,marker_num_2+1)], ignore_index=True)
        df_2_y = pd.concat([pd.DataFrame([root[1][2][i][1].text], columns=['y']) for i in range(1,marker_num_2+1)], ignore_index=True)

        #merge x and y into one df
        green_coord = pd.concat([df_1_x, df_1_y], axis=1)
        red_coord = pd.concat([df_2_x, df_2_y], axis=1)

        #save to csv
        green_coord.to_csv(filename.replace("\\","/").rsplit('/',1)[0]+'/'+(root[0][0].text).split('.')[0]+'_green.csv')
        red_coord.to_csv(filename.replace("\\","/").rsplit('/',1)[0]+'/'+(root[0][0].text).split('.')[0]+'_red.csv')