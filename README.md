# AI_Chat_Bot-Festus-Virtual-Assistant

**Instructions on common runtime errors** :page_with_curl:

Running the bot can be hindered by microsoft licensing whereby you will have to change the execution policy to AllSigned and later revert to RemoteSigned or according to your license reference.

NB: Updates of optimizers and libraries should be conducted after every release to avoid deprecations or malfunctioning.


#FILE DESCRIPTION 📁


* * **1. User Intentions**
* * [intents.json](./intents.json): This file contains the user intents which will be useed by the ANN to give out the respective response.
* 
* * **2. Training the ChatBot**
* * [training.py](./training.py): this file contains the script that trains the ANN with the data from the intents.json file.
* 
* * **3. ChatBot Modelling**
* * [chatbot.py](./chatbot.py): This file contains the final stages and scripts that model the bot from the generated pickle files and documents from the training stage.


**INSTALLATION
pip install -r requirements.txt
