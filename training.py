#importing relevant libraries for the bot training
import random
import json
import numpy as np
import pickle 
#importing nltk library
import nltk
from nltk.stem import WordNetLematizer
#impoting tensorflow for the machine learning algorithms
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation,Dropout
from tensorflow.keras.optimizers import SGD

#calling the WordNetLematizer constructor
lematizer=WordNetLematizer()
#loading the json file to develop our training
intentions=json.load(open('intentions.json').read())

#creating a list of words,classes and the document for tokenized words for training the bot
words=[]
classes=[]
document=[]
letters_to_be_ignored=['!','@','#','$','%','&','^','*','_','?','/',':',';','<','>','.','"']

#iterating through the intentions json file to tokenize the components of the file
for intention in intentions['intentions']:
    for pattern in intention['patterns']:
        #tokenizing the list of parameters parsed to the pattern list in json file
        word_list=nltk.wordtokenize(pattern)
        #instead of appending to the list to avoid duplication of data,lets extend the pattern list to a tokenized list pattern
        words.extend(word_list)
        #appending the tokenized word_list to the document list
        document.append(word_list,intention['tag'])
        if intention['tag'] not in classes:
            classes.append(intention['tag'])
print(document)#output of tokenized words is printed here according to the your json intention file



words = [lematizer.lematize(word) for word in words if word not in letters_to_be_ignored]
#removing duplicates using the sorted keyword and set constructor
words =sorted(set(words))
classes=sorted(set(classes))
print(words)#prints all the lematized words
print(classes)#prints all the lematized classes

#now saving the lematized words and classes to a pickle file
pickle.dump(words,open('words.pkl','wb'))#saving as words.pkl and mode is word binary mode
pickle.dump(classes,open('classes.pkl','wb'))#saving as classes and 


#Moving to the Machine Learning part now!!!!!!!

#since a machine only understands binaries(0s and 1s),lets create a bag of words containing binaries for our bot
training=[]
empty_output=[0]*len(classes)

#iterating through the training data(the document)

for dock in document:
    #creating an empty bag list
    word_bag =[]
    word_pattern=document[0]
    #lematizing the parsed word pattern
    word_pattern=[lematizer.lematize(word.lower()) for word in word_pattern]
    #invoking the ignored letters at this point isn't bad though for me,I don't see a big deal excluding them
    
    #appending the words into the word_bag
    for word in words:
        #for a word in words detected,a 1 is appended to the word_bag
        word_bag.append(1) if word in word_pattern else word_bag.append(0)
        #else the,word_bag is populated with a zero for the word not in words
    
#copying the list to an output row
row_out = list(empty_output)
row_out[classes.index(dock(1))]=1
#appending the word_bag and row_out to a training list
training.append(word_bag,row_out)


#FINAL STEP OF PREPROCESSING DATA

#shuffling and randomizing the training data
random.shuffle(training)
#turning the shuffled data into an array of data
training=np.array(training)
#splitting to x ana y values
x_train=list(training[:0])#for the 0 dimension data
y_train=list(training[:1])#for the 1 dimension data


#BUILDING THE NEURAL NETWORK
model =Sequential()
model.add(Dense(128,input_shape=(len(x_train[0]),),activation='relu'))
#relu is rectify linear unit


#preventing overfeeding of data to the model
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
#using the softmax feature allows to sum up our data and outputs the probability if getting a value
model.add(Dense(len(y_train[0]),activation='softmax'))

#using the SGD to instantiate the learning rate of the model
sgd=SGD(lr=0.01,decay=1e-6,momentum=0.9,nesterov=True)


#compiling the model
model.compile(loss='categorical crossentropy',optimizer=sgd,metric=['accuracy'])


hist=model.fit(np.array(x_train),np.array(y_train),epochs=200,batch_size=5,verbose=1)
#saving the model
model.save('Virtual_Festus_model.h5',hist)
print("WE ARE ALMOST THERE,KEEP ON GRINDING")
print("Done!")
