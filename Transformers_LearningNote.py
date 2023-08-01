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




