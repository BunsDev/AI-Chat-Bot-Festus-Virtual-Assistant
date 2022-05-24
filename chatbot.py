#importing relevant libraries for the bot development 
import random
import json
#import numpy as np
import pickle 
#importing nltk library
#import nltk
from nltk.stem import WordNetLematizer
import tensorflow
from tensorflow.keras.models import load_model

#calling the WordNetLematizer constructor
lematizer=WordNetLematizer()
#loading the json file to develop our bot
intentions=json.load(open('intentions.json').read())

#loading the pickle words and classes to develop our bot
words=pickle.load(open('words.pkl','rb'))#rb is for read binary mode
classes=pickle.load(open('classes.pkl','rb'))

#now loading the model
model=load_model('Virtual_Festus_model.h5')
#since the model will output binaries instead of words,lets convert the binaries to words


#first,lets clean the data 
def clean_data(sentences):
    word_sentence =nltk.wordtokenize(sentences)
    word_sentence=[lematizer.lematize(word) for word in word_sentence]
    return word_sentence


#second,convert the sentence into a bag of words using 1s and 0s(the binaries)
def bag_of_words(sentence):
    word_sentence =clean_data(sentence)
    bag=[0]*len(words)
    for z in word_sentence:
        for i,word in enumerate(words):
            if word==z:
                bag[i]=1
            else:
                bag[i]=0
                return np.array(bag)
            
            
#lastly,for predicting the classes,we do the following
def class_prediction(sentence):
    BAGGED_WORDS=bag_of_words(sentence)
    inter_result=model.predict(np.array(BAGGED_WORDS))[0]
    #LETS SET THE ERROR_THRESHOLD TO AROUND 25%
    ERROR_THRESHOLD=0.25
    final_result=[[i,r] for i,r in enumerate(inter_result)if r>ERROR_THRESHOLD]
    final_result.sort(key=lambda x: x[1],reverse=True)
    
    list_to_be_returned=[]
    for r in final_result:
        list_to_be_returned.append({'intentions':classes[r[0]], 'probability':str(r[1])})
        return list_to_be_returned
#and its done for full testing and integration 
#any feature for update contact +254796158635 or pull a request at festusmaithya.github.com