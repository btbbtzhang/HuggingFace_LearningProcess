# HuggingFace_LearningProcess
Focus on learning the Transformers from https://huggingface.co/docs/transformers/index 
Transformers aims on NLP (main focus), computer vision, computer vision, audio, mutimodal. Its architecutre looks like this:  
![Screen Shot 2023-08-01 at 1 50 47 PM](https://github.com/btbbtzhang/HuggingFace_LearningProcess/assets/34163897/fc0f2396-98dd-4b30-9884-f10c7b069bed)


Including 6 basic steps:
1. pieline -> 2. model/tokenizer -> 3. pytorch/tensorflow application -> 4. save/load -> 5. model hub (different model application, e.g., Bert,GPT, ...) -> finetune

## Pineline  
The most basic object in the Transformers library is the pipeline() function, allowing us to directly input any text and get an intelligible answer (NLP mainly).  

Some of the currently available pipelines are:  
feature-extraction (get the vector representation of a text)  
fill-mask (The idea of this task is to fill in the blanks in a given text:)  
question-answering  
sentiment-analysis  
summarization  
text-generation  
translation  
zero-shot-classification (unlabeled input text)  

## model & tokenizing 
model includes the architeture and checkpoint from different foundation model (pretrained on large-scale data)
convert text or other format of information into the numbers information (better for computer to understand)  

## steplize into details by using pytorch
using model, tokenizer and standard pytorch framework can have the same effect of using pipeline function from Transformers  

 
## Save & Load
The AutoModel class and all of its relatives are actually simple wrappers over the wide variety of models available in the library. It’s a clever wrapper as it can automatically guess the appropriate model architecture for your checkpoint, and then instantiates a model with this architecture.  

However, if you know the type of model you want to use, you can use the class that defines its architecture directly. Let’s take a look at how this works with a BERT model.  

![Screen Shot 2023-08-01 at 4 21 03 PM](https://github.com/btbbtzhang/HuggingFace_LearningProcess/assets/34163897/69385e04-bd36-4a8f-82ca-36c393543b2b)  

## Model hub
Different pretrained databases and models  

## Finetune
Using the Transformers on different specific datasets
1. tokenize your specific data into numeric information based on the existing foundation model (checkpoint from model)
   There is a way to speed up the tokenizing step by using Dataset.map() method.
   def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)
   tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
2. during the tokenize, the map() method can speed up the loading time, and batch the data into different smaller size groups, now we need a padding function can dynamically pad each batch instead of the entire dataset. Here the function DataCollatorWithPadding could be used to realize it.
   from transformers import DataCollatorWithPadding
   data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
   Here is the detailed example:

   ![Screen Shot 2023-08-02 at 10 09 20 AM](https://github.com/btbbtzhang/HuggingFace_TransformersTest/assets/34163897/d249d400-bab3-4112-8895-12ab1f6951a0)

3. Training
