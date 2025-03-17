import torch
from transformers import pipeline,BitsAndBytesConfig,AutoTokenizer,AutoModelForCausalLM,Trainer,TrainingArguments
from peft import LoraConfig, PeftModel
from trl import SFTConfig, SFTTrainer
from datasets import Dataset
import huggingface_hub
import pandas as pd
import os
import re
import gc

class CharacterChatbot():
    def __init__(self,model_path,dataset_path='/content/data/naruto_transcript.csv',Huggingface_token=None):
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.Huggingface_token = Huggingface_token
        self.base_model = 'meta-llama/Meta-Llama-3-8B-Instruct'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if Huggingface_token:
            huggingface_hub.login(token=self.Huggingface_token)
            if huggingface_hub.repo_exists(self.model_path):
                self.model = self.loadModel(self.model_path)
            else:
                print('model not found in hub. Proceeding with training ...')
                data = self.loadDataset(dataset_path)

                self.trainModel(self.base_model,data)

                self.model = self.loadModel(self.model_path)


    def filter_transcript(text):
        result = re.sub(r'\(.*?\)','',text) #replacing anything inside bracket in text string
        return result

    def loadDataset(self):
        transcript_df = pd.read_csv(self.dataset_path)
        transcript_df = transcript_df.dropna()
        transcript_df['line'] = transcript_df['line'].apply(self.filter_transcript)
        transcript_df['num_words'] = transcript_df['line'].apply(lambda x: len(x.split()))
        transcript_df['naruto_flag']=0
        transcript_df.loc[(transcript_df['name']=='Naruto') & (transcript_df['num_words']>5),'naruto_flag']= 1
        
        select_index = list(transcript_df.iloc[1:][transcript_df['naruto_flag']==1].index)  #take only naruto replies. So ignore starting index 0

        system_promt = """ You are Naruto Uzumaki from anime "Naruto". Your response should reflect his personality and speech patterns \n """

        prompts = []
        for i in select_index:
            prompt = system_promt
            prompt += transcript_df.iloc[i-1]['line']
            prompt += '\n'
            prompt += transcript_df.iloc[i]['line']
            prompts.append(prompt)

        df = pd.DataFrame({'prompt':prompts})

        dataset = Dataset.from_pandas(df)
        return dataset
    

    def trainModel(self,base_model,data,op_dir="./Chatbot_results"):
        
        bnbConfig = BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.float16)    #will make model weights in 4bits, using 'nf4' algo. and has compute_dtype (datatype in computation) is float16
        

        model =  AutoModelForCausalLM.from_pretrained(base_model,quantization_config=bnbConfig,trust_remote_code=True)
        model.config.use_cache = False

        tokenizer = AutoTokenizer.from_pretrained(base_model)
        tokenizer.pad_token = tokenizer.eos_token

        #LoRA config
        peft_config = LoraConfig(lora_alpha = 16, lora_dropout = 0.1, r = 64, bias="none", task_type="CASUAL_LM")

        training_args = SFTConfig(
            output_dir=op_dir,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            optim = 'paged_adamw_32bit', #paged for reduced memory consumption
            save_steps=200,
            logging_steps=10,
            learning_rate=2e-4,
            fp16=True,  #make floating point 16 to fit into memory
            max_grad_norm=0.3,
            max_steps=300,
            warmup_ratio=0.3,
            group_by_length=True,   #to make training more optimized -> as similar length closer to each other
            lr_scheduler_type='constant',
            report_to="none"    #can report to W&B
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=data,
            args=training_args,
            peft_config=peft_config,
            dataset_text_field = 'prompt',
            max_seq_len = 512  
        )

        trainer.train()

        #save
        trainer.model.save_pretrained("model_final")
        tokenizer.save_pretrained("tokenizer_final")

        #flush memory
        del trainer, model
        gc.collect()

        #
        BaseModel = AutoModelForCausalLM.from_pretrained(base_model,return_dict=True,quantization_config=bnbConfig,torch_dtype=torch.float16,device_map=self.device)

        tokenizer = AutoTokenizer.from_pretrained(base_model)

        model = PeftModel.from_pretrained(BaseModel,"model_final")
        model.push_to_hub(self.model_path)
        tokenizer.push_to_hub(self.model_path)

        #flush memory
        del BaseModel, model
        gc.collect()


    def loadModel(self,model_path):
        bnbConfig = BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.float16)    #will make model weights in 4bits, using 'nf4' algo. and has compute_dtype (datatype in computation) is float16
        model = pipeline('text-generation',model=model_path,model_kwargs={'torch_dtype':torch.float16,'quantization_config':bnbConfig})
        return model


    def model_chat(self,message,history):
        messages = []
        #system prompt
        messages.append(""" You are Naruto Uzumaki from anime "Naruto". Your response should reflect his personality and speech patterns \n """)

        for msg_n_resp in history:  #history is a list of list(w 2 strings)
            messages.append({'role':'user', 'content':msg_n_resp[0]})
            messages.append({'role':'assistant','content':msg_n_resp[1]})

        messages.append({'role':'user','content':message})

        terminator = [self.model.tokenizer.eos_token_id,
                      self.model.tokenizer.eos_tokens_to_ids("<|eot_id|>")]
        
        output = self.model(messages,max_length=256,eos_token_id=terminator,do_sample=True,temperature=0.6,top_p=0.9)

        output_msg = output[0]['generated_text'][-1]
        
        return output_msg