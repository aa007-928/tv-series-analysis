import pandas as pd
import os
from glob import glob

def load_subtitles_dataset(dataset_path):
    # files = glob(dataset_path+'/*.ass')
    files = glob(os.path.join(dataset_path,'*.ass'))
    scripts = []
    episode_nums = []
    for file in files:
        with open(file,'r',encoding='utf-8') as f:
            lines = f.readlines()
            lines = lines[27:]
            lines = [",".join(l.split(',')[9:]) for l in lines]
        lines = [l.replace('\\N',' ') for l in lines]
        script = " ".join(lines)
        episode_num = int(file.split('-')[-1].split('.')[0].strip())
        scripts.append(script)
        episode_nums.append(episode_num)

    df = pd.DataFrame({'episode':episode_nums, 'script':scripts})

    return df