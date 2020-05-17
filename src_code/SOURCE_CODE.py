import camelot
import pandas as pd
import os
import errno
import csv
import shutil

tables = camelot.read_pdf("ACUTE_INFARCTS_Updated.pdf",pages="all")

tables[1]

tables[0]

first_table = tables[0]

first_table.df

second_table = tables[1]

second_table.df

for table in tables:
    print(table.df)

for table in tables:
    table.to_excel("ACUTE_INFARCTS_Updated_3.xlsx")

df1 = pd.read_csv("ACUTE_INFARCTS_Updated_1.csv")

my_columns = df1["Location of acute infarct"]
print(my_columns)

os.mkdir('C:\\Users\\sonuk\\Desktop\\camelot_extraction\\CLEAN_DATA')

rootpath = 'C:\\Users\\sonuk\\Desktop\\camelot_extraction\\CLEAN_DATA'
for folders in my_columns:
    g = folders
    print(g)
    try:
        os.mkdir('C:\\Users\\sonuk\\Desktop\\camelot_extraction\\CLEAN_DATA\\'+folders)
    except OSError as e:
        if e.errno == errno.EEXIST:
            print("Nothing created")
        else:
            raise

col_Names=["Sl.", "Name", "Age/Sex", "", "Patient ID","Location of acute infract"]
df2= pd.read_csv("ACUTE_INFARCTS_Updated_2.csv",names=col_Names)
print(df2)

csvfile=open("ACUTE_INFARCTS_Updated_2.csv",'r')

c = csv.reader(csvfile)

lis=[]
for row in c:
    lis.append(row[5])
    
set_list=set(lis)
for i in set_list:
    try:
        os.mkdir('C:\\Users\\sonuk\\Desktop\\camelot_extraction\\CLEAN_DATA\\'+i)
    except OSError as e:
        if e.errno == errno.EEXIST:
            print("Nothing created")
        else:
            raise
    
df3 = pd.read_csv("ACUTE_INFARCTS_Updated_1.csv")
col_Names=["Sl.", "Name", "Age/Sex", "", "Patient ID","Location of acute infarct"]
df4= pd.read_csv("ACUTE_INFARCTS_Updated_2.csv",names=col_Names)
dicts={}
lis1=[i for i in df3['Location of acute infarct']]
print(lis1)
lis2=[i for i in df4['Location of acute infarct']]
print(lis2)

length1=len(lis1)
length2=len(lis2)
for i in range(length1):
    dest=lis1[i]
    source='/Case '+str(i+1)
    src_path='C:/Users/sonuk/Desktop/camelot_extraction/Assignment_2'+source
    dst_path='C:/Users/sonuk/Desktop/camelot_extraction/CLEAN_DATA/'+dest
    source_a = src_path +'/ADC.jpg'
    source_b = src_path +'/DWI.jpg'
    source_c = src_path +'/FLAIR.jpg'
    final_dest_a = dst_path +'/ADC.jpg'
    final_dest_b = dst_path +'/DWI.jpg'
    final_dest_c = dst_path +'/FLAIR.jpg'
    patha = shutil.copy(source_a,final_dest_a)
    pathb = shutil.copy(source_b,final_dest_b)
    pathc = shutil.copy(source_c,final_dest_c)
for i in range(length2):
    dest=lis2[i]
    source='/Case '+str(i+30)
    source_path='C:/Users/sonuk/Desktop/camelot_extraction/Assignment_2'+source
    dest_path='C:/Users/sonuk/Desktop/camelot_extraction/CLEAN_DATA/'+dest
    source_a = src_path +'/ADC.jpg'
    source_b = src_path +'/DWI.jpg'
    source_c = src_path +'/FLAIR.jpg'
    final_dest_a = dst_path +'/ADC.jpg'
    final_dest_b = dst_path +'/DWI.jpg'
    final_dest_c = dst_path +'/FLAIR.jpg'
    path1 = shutil.copy(source_a,final_dest_a)
    path2 = shutil.copy(source_b,final_dest_b)
    path3 = shutil.copy(source_c,final_dest_c)
