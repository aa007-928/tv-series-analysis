import torch
from transformers import pipeline
from nltk import sent_tokenize
import nltk
import numpy as np
import pandas as pd
import os
import sys
sys.path.append('../')
from utils import load_subtitles_dataset
nltk.download('punkt')
nltk.download('punkt_tab')


class themeClassifier():
    def __init__(self,theme_list,model='facebook/bart-large-mnli'):
        self.theme_list = theme_list
        self.model = model
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        self.classifier = pipeline('zero-shot-classification',model=self.model,device=self.device)


    def get_inference(self,script):
        script_sentences = sent_tokenize(script)
        #batch sentence
        sent_batch_size = 20
        script_batches_list = []
        for i in range(0,len(script_sentences),sent_batch_size):
            sent = " ".join(script_sentences[i:i+sent_batch_size])
            script_batches_list.append(sent)
        #run model
        theme_op = self.classifier(script_batches_list,self.theme_list,multi_label=True)    #use a smaller batch of script_batch_list on cpu like [:2]
        # wrangle output
        themes = {}
        for op in theme_op:
            for label,score in zip(op['labels'],op['scores']):
                if label not in themes:
                    themes[label] = []
                themes[label].append(score)
        themes = {key:np.mean(np.array(val)) for key,val in themes.items()}
        return themes
    
    def get_themes(self,dataset_path,save_path=None):
        #read from prev saved op if available
        if save_path and os.path.exists(save_path):
            dataset_df = pd.read_csv(save_path)
            return dataset_df

        #load dataset
        dataset_df = load_subtitles_dataset(dataset_path)
        # dataset_df = dataset_df.head(2)
        #run inference
        inference_df = dataset_df['script'].apply(self.get_inference)
        
        themes_df = pd.DataFrame(inference_df.tolist())
        dataset_df[themes_df.columns] = themes_df

        #save
        if save_path:
            dataset_df.to_csv(save_path,index=False)
        
        return dataset_df