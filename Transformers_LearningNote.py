# installations
# for M1 users, get the following installed before installing TensorFlow or pytorch
# MPS acceleration is available on MacOS 12.3+:  pip3 install torch torchvision torchaudio
# brew install cmake brew install pkg-config
# pip install transformers

from transformers import pipeline 
classifier = pipeline("sentiment-analysis")
res = classifier("I am happy to test on Transformers library from HuggingFace.")
print(res)
# [{'label': 'POSITIVE', 'score': 0.9992256164550781}]


classifier = pipeline("zero-shot-classification")
result = classifier(
    "This is a course about the Transformers library",
    candidate_labels=["education", "politics", "business"],
)
print(result)
#
#{'sequence': 'This is a course about the Transformers library',
# 'labels': ['education', 'business', 'politics'],
# 'scores': [0.8445963859558105, 0.111976258456707, 0.043427448719739914]}

generator = pipeline("text-generation", model = "distilgpt2")
res = generator(
  "In this course, we will learn how to",
  max_lenghth = 33,
  num_return_sequences = 2,
)
#
# Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
# [{'generated_text': 'In this course, we will teach you how to practice the fundamentals in this seminar. It will take you through the process and teach you the techniques required in these classes'},
# {'generated_text': "In this course, we will teach you how to think differently about the consequences of an economic failure. In fact, we've been told by many of our teachers to"}]

## showing the funcionality of tokenizer (tokenizers need to convert our text inputs to numerical data.)
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel 
#(this can be other classes for a specific purpose)
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english" # this is the same for pipeline("sentiment-analysis")
model = AutoModelForSequenceClassification.from_pretrained(checkpoint) # this is equal to model = AutoModel(checkpoint) (this model is usually used for general case)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

classifier = pipeline("sentiment-analysis", model = model, okenizer = tokenizer)
seq = "testing the tokenizer function"
res = tokenizer(seq, padding = T, truncation = T, return_tensors = "pt")
print(res)
outputs = model(**res)
print(outputs.last_hidden_state.shape)
tokens = tokenizer.tokenize(seq)
print(tokens)
# attention_mask is for attention layer that 0 means layer should ignore it
ids = tokenizer.convert_tokens_to_ids(okens)
print(ids)
decoded_string = tokenizer.decode(ids)
print(decoded_string)



#### Transformers used and compared to pytorch
import torch
import torch.nn.functional as F

x_train = ["I've been waiting for a HuggingFace course my whole life.", "Python is great!"]
res = classifier(x_train)
print(res)

## doing the same as the above for the detailed steps in pytorch standard model, which could be used to finetune our model pytorch training loop

batch = tokenizer(x_train, padding=True,truncation=T,max_length=512,return_tensors="pt") # eturn_tensors="pt": pytorch format
print(batch)

with torch.no_grad():
    outputs = model(**batch) #batch is a dictionary
    print(outputs)
    predictions = F.softmax(outputs.logits, dim = 1)
    print(predictions)
    labels = torch.argmax(predictions, dim=1)
    print(labels)


### Saving and Loading
# save the tokens and model
save_path = "saved"
tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)

# load the tokens and model
tok = AutoToeknizer.from_pretrained(save_path)
mod = AutoModelForSequenceClassification.from_pretrained(save_path)


## model hub (using different model for different purposes)


### Finetune our own model
# 1. prpare dataset
# 2. load pretrained tokenizer, call it with dataset -> encoding
# 3. build pytorch dataset with encodings
# 4. load pretrained model
# 5. a). load trainer and train int
# #  b). native pytorch trainning loop

from transformers import Trainer, TrainingAruments
training_args = TrainingAruments("test-trainer")

trainer = Trainer(
    model,
    training_args,
    train_dataset = tokenized_datasets["train"],
    eval_dataset = tokenized_datasets["validation"],
    data_collator = data_collator,
    tokenizer = tokenizer,
)

trainer.train()
