# import sys
# from PyQt5.QtWidgets import QApplication, QWidget, QLabel
# from PyQt5.QtGui import QIcon
# from PyQt5.QtCore import pyqtSlot
#
# def window():
#    app = QApplication(sys.argv)
#    widget = QWidget()
#
#    textLabel = QLabel(widget)
#    textLabel.setText("Hello World!")
#    textLabel.move(110,85)
#
#    widget.setGeometry(50,50,320,200)
#    widget.setWindowTitle("PyQt5 Example")
#    widget.show()
#    sys.exit(app.exec_())
#
# if __name__ == '__main__':
#    window()



import remi.gui as gui
from remi import start, App
from threading import Timer

# from transformers import pipeline
# classifier = pipeline("sentiment-analysis")
# res = classifier("We are very happy to see you!")
# print(res)

#-------------data clean---------------------
#----------for train data clean--------------------
import csv
import os
#
# f=open("data")
# line=f.readline()
# with open("train.csv", 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     #writer.writerow(["text", "label"])
#     lb=0
#     while line:
#         #print(line)
#         d1=line[9]
#         d2=line[12:]
#         if d1=="2":
#             writer.writerow([1, d2]) #positive
#         if d1=="1":
#             writer.writerow([0, d2]) #negative
#         line = f.readline()
# f.close()
#--------------------------------------------------------
#----------for test data clean--------------------
# f=open("test.txt")
# line=f.readline()
# with open("test.csv", 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     # writer.writerow(["text", "label"])
#     lb=0
#     while line:
#         #print(line)
#         d1=line[9]
#         d2=line[12:]
#         if d1=="2":
#             writer.writerow([1,d2]) #positive
#         if d1=="1":
#             writer.writerow([0,d2]) #negative
#         line = f.readline()
# f.close()
#--------------------------------------------------------


# from transformers import pipeline
# from transformers import AutoTokenizer,AutoModelForSequenceClassification
#
# model_name="ydshieh/tiny-random-gptj-for-sequence-classification" #"distilbert-base-uncased-finetuned-sst-2-english"
# model=AutoModelForSequenceClassification.from_pretrained(model_name)
# tokenizer=AutoTokenizer.from_pretrained(model_name)
# classifier=pipeline("sentiment-analysis",model=model,tokenizer=tokenizer)
# res=classifier("we love the show")
# print(res)

# tokens=tokenizer.tokenize("we love the show")
# token_ids=tokenizer.convert_tokens_to_ids(tokens)
# input_ids=tokenizer("we love the show")
#
# print(f'    Tokens: {tokens}')
# print(f'Tokens IDs: {token_ids}')
# print(f' Input IDs: {input_ids}')

#-----------------------------------------------
#--------------fine tunning----------------------
from pathlib import Path
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments

# model_name="distilbert-base-uncased"
#-------------below is example--------------------------------

#
# def read_imdb_split(split_dir):
#     split_dir = Path(split_dir)
#     texts = []
#     labels = []
#     for label_dir in ["pos", "neg"]:
#         for text_file in (split_dir/label_dir).iterdir():
#             texts.append(text_file.read_text())
#             labels.append(0 if label_dir == "neg" else 1)
#     return texts, labels
#
# train_texts, train_labels = read_imdb_split('aclImdb/train')
# test_texts, test_labels = read_imdb_split('aclImdb/test')
#------------------example end----------------------------

# def read_split(filename):
#     f=open(filename)
#     line=f.readline()
#     texts = []
#     labels = []
#     while line:
#         print(line)
#         d1=line[9]
#         d2=line[12:]
#         texts.append(d2)
#         if d1=="2":
#             labels.append(1) #positive
#         if d1=="1":
#             labels.append(0) #negative
#         line = f.readline()
#     f.close()
#     return texts, labels
#
# train_texts, train_labels = read_split('test.txt')
# test_texts, test_labels = read_split('test.txt')
#
# train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2)
#
# class IMDbDataset(Dataset):
#     def __init__(self, encodings, labels):
#         self.encodings = encodings
#         self.labels = labels
#
#     def __getitem__(self, idx):
#         item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
#         item['labels'] = torch.tensor(self.labels[idx])
#         return item
#
#     def __len__(self):
#         return len(self.labels)
#
# tokenizer=DistilBertTokenizerFast.from_pretrained(model_name)
#
# train_encodings=tokenizer(train_texts,truncation=True,padding=True)
# val_encodings=tokenizer(val_texts,truncation=True,padding=True)
# test_encodings=tokenizer(test_texts,truncation=True,padding=True)
#
# train_dataset = IMDbDataset(train_encodings, train_labels)
# val_dataset = IMDbDataset(val_encodings, val_labels)
# test_dataset = IMDbDataset(test_encodings, test_labels)
#
# training_args=TrainingArguments(
#     output_dir='./results',
#     num_train_epochs=2,
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=64,
#     warmup_steps=500,
#     learning_rate=5e-5,
#     weight_decay=0.01,
#     logging_dir='./logs',
#     logging_steps=10,
# )
#
# model=DistilBertForSequenceClassification.from_pretrained(model_name)
#
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=val_dataset
# )
#
# trainer.train()
# trainer.evaluate()
#
# model_dir = '/PycharmProjects/pythonProject/trainedmodel/'
# trainer.save_model(model_dir + 'fine-tunned-model')


#----------test gpt neo colab data prepare------------
import os
from pathlib import Path
f=open("test.txt")
line=f.readline()
n=0
while line:
    print(line)
    tmp = str(n)
    if line[9] == '2':
        p = os.getcwd()
        p = p + '/test/pos/' + tmp + '.txt'
        with open(p,'a') as f1:
            f1.write(line[11:-1])
    if line[9] == '1':
        p = os.getcwd()
        p = p + '/test/neg/' + tmp + '.txt'
        with open(p,'a') as f1:
            f1.write(line[11:-1])
    line = f.readline()
    n=n+1
f.close()
#--------------------------------------------------------
