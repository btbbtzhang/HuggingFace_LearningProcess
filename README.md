# HuggingFace_LearningProcess
Focus on learning the Transformers from https://huggingface.co/docs/transformers/index 
Transformers aims on NLP (main focus), computer vision, computer vision, audio, mutimodal
Including 6 basic steps:
1. pieline -> 2. model/tokenizer -> 3. pytorch/tensorflow application -> 4. save/load -> 5. model hub (different model application, e.g., Bert,GPT, ...) -> finetune

## Pineline  
The most basic object in the Transformers library is the pipeline() function, allowing us to directly input any text and get an intelligible answer (NLP mainly).  

Some of the currently available pipelines are:  
feature-extraction (get the vector representation of a text)
fill-mask
question-answering
sentiment-analysis
summarization
text-generation
translation
zero-shot-classification (unlabeled input text)

