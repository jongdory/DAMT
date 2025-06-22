import os
import re
import csv
import pandas as pd

def sentence_to_dict(sentence, prefix):
    result = {}
    for i, line in enumerate(sentence.split("\n")[:-1]):
        elements = line.split(", ")
        if i ==3: break
        key = elements[2]
        value = float(elements[3])
        result[f"{prefix}_{key}"] = value
    return result

def sentence_to_dict2(sentence, prefix):
    headers = sentence.split("\n")[0].split(" ")[:19]
    data = sentence.split("\n")[1:]
    data_dict = {}
    for row in data[:-1]:
        row_data = re.split(r'\s+', row)
        for i, header in enumerate(headers[3:10]):
            data_dict[f"{prefix}_{row_data[0]}_{header}"] = float(row_data[i+1].strip())
        
    return data_dict

def file_to_dict(path, prefix):
    with open(path, "r") as file:
        contents = file.read()
    
    result = {}
    sentence = contents[contents.find("# Measure Cortex, NumVert,"):contents.find("# NTableCols 10")]
    result.update(sentence_to_dict(sentence, prefix))
    sentence2 = contents[contents.find("# ColHeaders"):]
    result.update(sentence_to_dict2(sentence2, prefix))
    
    return result

if __name__ == "__main__":
    data_path = "OASIS/OASIS3"

    flag = True
    with open('feats_global.csv','w') as f:
            w = csv.writer(f)
            mica_path = f"{data_path}/output/micapipe"
            free_path = f"{data_path}/output/freesurfer"
            for subject in os.listdir(mica_path):
                if subject[:3] != "sub": continue
                mica_sub_path = os.path.join(mica_path, subject)
                if os.path.exists(mica_sub_path) == False: continue
                for session in os.listdir(mica_sub_path):
                    if session == "ses-M00" or session == "ses-00":
                        free_sub_path = os.path.join(free_path, f"{subject}_{session}")
                        lstat_path = os.path.join(free_sub_path, "stats/lh.aparc.stats")
                        rstat_path = os.path.join(free_sub_path, "stats/rh.aparc.stats")
                        if os.path.exists(lstat_path) == False: continue
                        result = {"subject" : f"{subject}_{session}"}
                        result.update(file_to_dict(lstat_path,"l"))
                        result.update(file_to_dict(rstat_path,"r"))
                        if flag:
                            w.writerow(result.keys())
                            flag = False
                        w.writerow(result.values())
                

    df = pd.read_csv("feats_global.csv", index_col="subject")
    normalization_df = (df - df.mean())/df.std()
    normalization_df.head()
    normalization_df.to_csv("nfeats_global.csv")