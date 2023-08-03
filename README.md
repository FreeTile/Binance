!I am not responsible for your lost funds, follow further instructions at your own risk!

# Binance
Neural Network for trading

##### Notice
Sorry, there is a lot of junk in my code, it is my first attempt to create a working neural network.

## Overall
There are several scripts that can automatically buy and sell kryptocurrency wich use a neural network. All that you need are an API key and a API secret from your Binance account (you can find instructions on how to create it [here](https://www.binance.com/en/support/faq/how-to-create-api-keys-on-binance-360002502072)). This project already has one best individual for training and one trained model, so you can just run Trading.py with your API settings. More details about each script are can be found below the "Getting started" section.

## Getting started
There are two scripts, wich are called Average_shadows and LoadData. Average shadows calculate the average numbers between an open price and the lowest price and between the highest price and a close price or vice versa.
LoadData is self-explanatory, it's loads data :D and it converts it into the format that can be accepted by the neural network. Before you begin you need to compile "Average_Shadows" and "LoadData".
1. Before running any scripts you need to configure the file named "config.txt". In this file you can find parameters for all scripts from lines 2 to 5. Settings for the genetic algorithm are placed between lines 7 and 11, etc.
2. As for the genetic algorithm. This file finds an individual with the best parameters for training, and after that you can fit it and use it for your trading. *WARNING* If you have a weak PC, it can take a week or more to produce a more adapted individual. I rented a server with a powerful GPU with 32GB of VRAM, so you need to check all the parameters in "config.txt" like the number of generations or the population size. Also if you want to have many diverse individuals, you can increase the mutation rate parameter.
3. Now that you have the best individual for training, you need to run "TrainBestModel.py" to train that model before you start the traiding. If you see that the loss parameter continues to go down at the end of the training, you can increase the number of epochs to get a better model.
4. Almost done! You have a traned model of a neural network that can predict next changes in a candlestick chart by analysing the previous 20 charts. You can run "Trading.py" to see how it works and if it can they predict the pattern. This will give you the information how profitable the model is.
   
## How it works

### Genetic Algorithm
This is the hardest script of this project (for you and for your PC :D ). In config.txt you can find settings for it at lines 7-10. We have numbers of generations and population size. The second mean how much model will be created at one generation. Every model has it own number of layers, types of layers, bathc size etc. After creating all models script looking for some best models using loss rate and accuracy rate. After this step it cross them together and some of crossed individuals are mutated. Rate of mutation also located in config.txt. 0.2 mean that individual will mutate in 20% cases. In the end of every population the best model saving in "models/best_individual.pkl", so if there was something wrong and your PC turned off, you will have the last best model. 
#### Notice
In this script epochs are static and its value is 15. In TrainBestModel.py number of epochs are 50, keep it in mind.

### Train Best Model
As you already have best model for traning, you can run this script. It takes data from train_data.npy and true values from train_labels.npy. Then it divide train_data and train_labels into blocks of batch size. It need to save RAM, because if you dont have anought RAM, there will be error. At the end it save trained model into "models/trained_model.h5".

### Trading
This is one of the main scripts. It connects to the Binance API using your API key and API secret. Then it start cicle
1. Close all open orders
2. Get 21 previous candles, the last candle still trading so we don't take it into prediction.
3. Neural Network give us a prediction of the next candle, will go it up or down
4. Using prediction script send request to buy or sell BTC, also it sell stop loss request in case if price go down instead up, so we won't lose a lot of money if it make a wrong prediction.
If there was several candels with one direction and the model predict that the next candle will go viceverse, first of all script will return all funds in state before running the script.
5. In the end it takes time before next cicle. As we trade with 5 min candles, it repeats every time the time is a multiple of 5 minutes.

### Average shadows
This script calculate average up and down shadows of every candel using train_data.npy. This parameter can be used for calculating stop loss prices in Trading.py. If you have train_data.npy, you can run it and remember average numbers of up and down shadows.

### LoadData
This script loads data by sending request using Binance API and save it to train_data.npy and train_labels.npy. Train_data is a massive of open, high, low, close prices, volume of tradings, close time, numbers of trades, Directional Movement index, momentum etc. It contain it in blocks each have 20 charts and for each this block there is we have Train_labels, which has true value for each block from Train_data. Using both files you can use Genetic Algorithm, TrainBestModel, Average Shadows and Trading scripts. This file downloads data for 5 years ago, you can change this parameter in config.txt, this parameter should be in days, not in years (maybe it can be used with years too, but I prefer use it writing days)
