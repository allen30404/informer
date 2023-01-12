import pandas as pd
import argparse
import os
import csv
from pandas._libs.tslibs.timestamps import Timestamp
import torch
parser = argparse.ArgumentParser(description='[Data_Process]')


parser.add_argument('--root_path', type=str, default='../data/OriginalData/', help='root path of the data file')
parser.add_argument('--data_after_process_path', type=str, default='../data/Data_0_24/', help='data file')  
parser.add_argument('--data_start', type=str, default='0', help='data time start')  
parser.add_argument('--data_end', type=str, default='23', help='data time end')  

args = parser.parse_args()

data_list = ['site_north_complete.csv'] #檔名
data_capacity = [907680] #裝置容量
data_dict = {
    'filename':data_list,
    'capacity':data_capacity
}
data_df = pd.DataFrame(data_dict)


for each_data in range(len(data_df)):
    
    table = []
    # 開啟 CSV 檔案
    with open(os.path.join(args.root_path,data_df.loc[each_data,'filename']), newline='') as csvfile:

        # 讀取 CSV 檔案內容
        rows = csv.reader(csvfile)
        #不儲存第一行
        first_row = False
        # 以迴圈輸出每一列
        for row in rows:
            if first_row:
                #日期時間小於10補0
                if int(row[1]) < 10 : row[1] = '0'+str(row[1])
                if int(row[2]) < 10 : row[2] = '0'+str(row[2])
                if int(row[3]) < 10 : row[3] = '0'+str(row[3])
                #決定那些feature寫入
                table.append([
                    ('date',row[0]+'-'+row[1]+'-'+row[2]+' '+row[3] +':00:00'),
                    ('NWP_Radiation',row[5]),
                    #('NWP_Rainfall',row[6]),
                    ('NWP_Temperature',row[7]),
                    #('NWP_Sealevelpressure',row[8]),
                    ('NWP_WindSpeed',row[9]),
                    #('NWP_Humidity',row[10]),
                    ('PVtemp',row[12]),
                    ('Irradiation',row[11]),
                    ('OT',row[4])])
            first_row = True


        list_of_dicts = [dict(x) for x in table]
        df_table = pd.DataFrame(list_of_dicts)
        cols = list(df_table.columns); cols.remove('date'); cols.remove('OT')
        max = df_table[cols].astype(float).max(axis=0)
        min = df_table[cols].astype(float).min(axis=0)
        max_min_sub = max.sub(min,axis=0)
        df_table = df_table[['date']+cols+['OT']]
    
        output = [
            [
            'date',
            'NWP_Radiation',
            #'NWP_Rainfall',
            'NWP_Temperature',
            #'NWP_Sealevelpressure',
            'NWP_WindSpeed',
            #'NWP_Humidity',
            'PVtemp',
            'Irradiation',
            'OT']] #feature名稱

        data_timestmap_24 = 0
        for i in range(len(df_table)-1):
            if i>=int(args.data_start)+data_timestmap_24 and i<=int(args.data_end)+data_timestmap_24:
                output.append(
                    [df_table['date'][i],
                    (float(df_table['NWP_Radiation'][i+1])-min['NWP_Radiation'])/max_min_sub['NWP_Radiation'],
                    #(float(df_table['NWP_Rainfall'][i+1])-min['NWP_Rainfall'])/max_min_sub['NWP_Rainfall'],
                    (float(df_table['NWP_Temperature'][i+1])-min['NWP_Temperature'])/max_min_sub['NWP_Temperature'],
                    #(float(df_table['NWP_Sealevelpressure'][i+1])-min['NWP_Sealevelpressure'])/max_min_sub['NWP_Sealevelpressure'],
                    (float(df_table['NWP_WindSpeed'][i+1])-min['NWP_WindSpeed'])/max_min_sub['NWP_WindSpeed'],
                    #(float(df_table['NWP_Humidity'][i+1])-min['NWP_Humidity'])/max_min_sub['NWP_Humidity'],
                    (float(df_table['PVtemp'][i+1])-min['PVtemp'])/max_min_sub['PVtemp'],
                    (float(df_table['Irradiation'][i])-min['Irradiation'])/max_min_sub['Irradiation'],
                    float(df_table['OT'][i])/data_df.loc[each_data,'capacity']])


            if i==int(args.data_end)+data_timestmap_24:
                data_timestmap_24 += 24




    with open(os.path.join(args.data_after_process_path,data_df.loc[each_data,'filename']), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # 寫入二維表格
        writer.writerows(output)


