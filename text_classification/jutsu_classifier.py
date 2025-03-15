import torch
import huggingface_hub
from transformers import AutoTokenizer, AutoModelForSequenceClassification,DataCollatorWithPadding,TrainingArguments,Trainer,pipeline
from datasets import Dataset
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import evaluate
import gc
from .cleaner import Cleaner
from sklearn.utils.class_weight import compute_class_weight
from .custom_trainer import customTrainer
from .training_utils import getClassWts,compute_metrics

class jutsuClassifier:
    def __intit__(self,saved_model_path,dataset_path=None,model_name="distilbert/distilbert-base-uncased",text_col='text',label_col='jutsu',test_size=0.2,num_labels=3,Huggingface_token=None):
        self.model_name = model_name
        self.saved_model_path = saved_model_path
        self.dataset_path = dataset_path
        self.text_col = text_col
        self.label_col = label_col
        self.test_size = test_size
        self.num_labels = num_labels
        self.Huggingface_token = Huggingface_token
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if self.Huggingface_token:
            huggingface_hub.login(token=self.Huggingface_token)

        self.tokenizer = self.loadTokenizer()

        if not huggingface_hub.repo_exists(self.saved_model_path):
            if not self.model:
                raise ValueError('model/saved_model path is required')
            
            data_train_tokenized, data_test_tokenized = self.loadDataset(self.dataset_path) #return tokenized 
            
            data_train = data_train_tokenized.to_pandas()
            data_test = data_test_tokenized.to_pandas()
            data = pd.concat([data_train,data_test]).reset_index(drop=True)
            # classWts = compute_class_weight("balanced", classes=sorted(data['label'].unique().tolist(), y=data['label'].tolist()))
            classWts = getClassWts(data)

            self.trainModel(data_train_tokenized,data_test_tokenized,classWts)

        self.model = self.loadModel(self.saved_model_path)

    
    def loadTokenizer(self):
        model_in_use = self.saved_model_path if huggingface_hub.repo_exists(self.saved_model_path) else self.model
        return AutoTokenizer.from_pretrained(model_in_use)
    
    def get_single_jutsu(self,jutsu):
        if 'Genjutsu' in jutsu:
            return 'Genjutsu'
        if 'Ninjutsu' in jutsu:
            return 'Ninjutsu'
        if 'Taijutsu' in jutsu:
            return 'Taijutsu'
        
    def preProcessing(self,tokenizer,exmaples):
        return tokenizer(exmaples['text'],truncation=True)
    
    def laodDataset(self,dataset_path):
        df = pd.read_json(dataset_path,lines=True)  #read from jsonl
        df['single_jutsu_type'] = df['jutsu_type'].apply(self.get_single_jutsu)
        df['text'] = df['jutsu_name'] + '. ' + df['jutsu_descp']
        df[self.label_col] = df['single_jutsu_type']
        df = df[['text',self.label_col]]
        df.dropna()

        #clean text
        cleaner = Cleaner()
        df['text'] = df[self.text_col].apply(cleaner.clean)

        #encoding label
        le = preprocessing.LabelEncoder()
        le.fit(df[self.label_col].tolist())
        self.label_dict = {i:label for i,label in enumerate(le.__dict__['classes_'])}
        df['label'] = le.transform(df[self.label_col].tolist())

        #split data into train & test
        data_train, data_test = train_test_split(df,test_size=self.test_size,stratify=df['label'])
        # #convert pandas df to Huggingface dataset
        data_train = Dataset.from_pandas(data_train)
        data_test =  Dataset.from_pandas(data_test)
        # #tokenize the dataset
        data_train_tokenized = data_train.map(lambda x: self.preProcessing(self.tokenizer,x), batched=True)
        data_test_tokenized = data_test.map(lambda x: self.preProcessing(self.tokenizer,x), batched=True)

        return data_train_tokenized,data_test_tokenized
        # return data_train, data_test


    def trainModel(self,data_train,data_test,classWts):
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name,num_labels=self.num_labels,id2label=self.label_dict)
        dataCollator = DataCollatorWithPadding(tokenizer=self.tokenizer)    #pads dataset

        training_args = TrainingArguments(
            output_dir=self.saved_model_path,
            learning_rate=2e-4,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_training_epochs = 5,
            weight_decay=0.01,
            evaluation_strategy='epoch',
            logging_strategy='epoch',
            push_to_hub=True    #model uploaded to huggingface_hub after training
        )

        trainer = customTrainer(model=model,args=training_args,train_dataset=data_train,eval_dataset=data_test,tokenizer=self.tokenizer,data_collator=dataCollator,compute_metrics=compute_metrics)

        trainer.set_device(self.device)
        trainer.set_class_weights(classWts)

        trainer.train()

        #flush memory (after training)
        del trainer,model
        gc.collect()

        if self.device=='cuda':
            torch.cuda.empty_cache()

    def loadModel(self,model_path):
        model = pipeline('text_classification',model=model_path,return_all_score=True)
        return model
    
    def postProcessing(self,model_op):
        op = []
        for pred in model_op:
            label = max(pred, key= lambda x: x['score'])['label']
            op.append(label)
        return op

    
    def classification_inference(self,text):
        model_op = self.model(text)
        preds = self.postProcessing(model_op)
        return preds

