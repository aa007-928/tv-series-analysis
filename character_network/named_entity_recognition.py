import spacy
from nltk import sent_tokenize
import pandas as pd
from ast import literal_eval
import os
import sys
sys.path.append('../')
from utils import load_subtitles_dataset

class named_entity_recog:
    def __init__(self,model="en_core_web_trf"):
        self.nlp_model = spacy.load(model)

    def get_ners_inference(self,script):
        sentences = sent_tokenize(script)
        ner_op = []
        for line in sentences:
            doc = self.nlp_model(line)
            ners = set()
            for entity in doc.ents:
                if entity.label_ == 'PERSON':
                    fullname = entity.text
                    firstname = fullname.split(" ")[0].strip()
                    ners.add(firstname)
            ner_op.append(ners)
        return ner_op

    def get_ners(self,dataset_path,save_path=None):
        if save_path and os.path.exists(save_path):
            df = pd.read_csv(save_path)
            df = df['ners'].apply(lambda x: literal_eval(x) if isinstance(x,str) else x) #list is stored as string in csv. Converted back to list
            return df
            
        df = load_subtitles_dataset(dataset_path)
        # df = df.head(10)    #limit for test on cpu
        df['ners'] = df['script'].apply(self.get_ners_inference)

        if save_path:
            df.to_csv(save_path,index=False)

        return df
    
    
