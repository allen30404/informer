import pandas as pd
import argparse
import os
import csv

parser = argparse.ArgumentParser(description='[Data_Combine]')


parser.add_argument('--root_path', type=str, default='./data/Data_0_24/', help='root path of the data file')
parser.add_argument('--data_after_process_path', type=str, default='./data/Data_0_24/', help='data file') 

args = parser.parse_args()


data_list = ['place_7.csv','place_9.csv'] #檔名

output = [[
    'date',
    'NWP_Radiation',
    'NWP_Rainfall',
    'NWP_Temperature',
    'NWP_Sealevelpressure',
    'NWP_WindSpeed',
    #'NWP_Humidity',
    'Irradiation',
    'OT']] #feature名稱

for each_data in data_list:
    first_row = False
    row_count = 0 #計算長度，將不滿24小時的資料刪除
    print(each_data)

    with open(os.path.join(args.root_path,each_data), newline='') as csvfile:

    # 讀取 CSV 檔案內容
        rows = csv.reader(csvfile)

        # 以迴圈輸出每一列
        for row in rows:
            if first_row == True and row_count <=3119:
                output.append(row)
                row_count += 1
            first_row = True


with open(os.path.join(args.data_after_process_path,'combine_data_chiayi.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # 寫入二維表格
        writer.writerows(output)
