The main discussion takes place on the Russian forum 2ch/ai/ https://2ch.hk/ai/res/385611.html#453778
My Telegram - https://t.me/JoJobBizzareAdventure

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
1. Before running any scripts you need to configure the file named "config.txt". In this file you can find parameters for all scripts from lines 1 to 19 and parameters for other scripts. 
2. As for the genetic algorithm. This file finds an individual with the best parameters for training, and after that you can fit it and use it for your trading. *WARNING* If you have a weak PC, it can take a week or more to produce a more adapted individual. I rented a server with a powerful GPU with 32GB of VRAM, so you need to check all the parameters in "config.txt" like the number of generations or the population size. Also if you want to have many diverse individuals, you can increase the mutation rate parameter.
3. Now that you have the best individual for training, you need to run "TrainBestModel.py" to train that model before you start the traiding. If you see that the loss parameter continues to go down at the end of the training, you can increase the number of epochs to get a better model.
4. Almost done! You have a traned model of a neural network that can predict next changes in a candlestick chart by analysing the previous 20 charts. You can run "Trading.py" to see how it works and if it can they predict the pattern. This will give you the information how profitable the model is.
   
## How it works

### Genetic Algorithm
This is the hardest script of this project (for you and for your PC :D ). In "config.txt" you can find settings for it between the lines 7-10. There we have the number of generations and the population size. The latter means how many models will be created in one generation. Every model has its own number of layers, types of layers, batch size, etc. After creating all models script looks for the best models using the loss rate and the accuracy rate of each model. After this step it crosses them together and some of the crossed individuals mutate. The mutation rate parameter is also located in "config.txt". 0.2 means that an individual will mutate in 20% of cases. In the end of every population the best model is saved in "models/best_individual.pkl", so if something goes wrong and your PC turns off, you will have the last best model saved. 
#### Notice
In this script epochs are static and their value is 15. In TrainBestModel.py the number of epochs is 50, keep that in mind.

### Train Best Model
Since you already have best model for traning, you can run this script. It takes data from "train_data.npy" and true values from "train_labels.npy". Then it divide "train_data" and "train_labels" into blocks, each blocks size is recorded in the "batch_size" parameter. It is necessary to reduce RAM usage, because if you don't have enough RAM, there will be an error. In the end it saves a trained model into "models/trained_model.h5".

### Trading
This is one of the main scripts. It connects to the Binance API using your API key and API secret. Then it starts the cycle:
1. Close all open orders
2. Get the previous 21 candlesticks (the candlestick wich is trading at the moment of the cycle run is not taken into consideration).
3. The neural network gives us a prediction whether the next candlestick will go up or down
4. Using the prediction the script sends a request to buy or sell BTC, also it sends a stop loss request in case the price go down instead of up, so we don't lose a lot of money if it makes an incorrect prediction.
If there were several candelsticks with the same direction and the model predicts that the next candlestick will go in the opposite direction, the script will return all the funds into the default state before making the next trade. (The default state is a state when 50% of balance is in USDT and 50% is in BTC, so the script can trade without problems)
5. The next cycle doesn't start immediately. Since the candelstick charts that the neural network uses refresh every 5 minutes, the cycle can be repeated only in a period of time divisible by 5.

### Average shadows
This script calculates the average upper and lower shadows of every candelstick using "train_data.npy". This parameter can be used for calculating stop loss prices in "Trading.py". If you have "train_data.npy", you can run it and remember average numbers of upper and lower shadows.

### LoadData
This script loads data by sending the request using Binance API and saves it to "data/train_data.npy" and "data/train_labels.npy". "Train_data" is an array of open, high, low, close prices, volume of tradings, closing times for each candelsticks, numbers of trades, Directional Movement index, momentum etc. It contains data in blocks, each block consists of 20 charts and for each block there are true values in "Train_labels". Using both files you can use "Genetic_Algorithm.py", "TrainBestModel.py", "Average_Shadows.py" and "Trading.py" scripts. This file downloads data for the last 5 years, and you can change this parameter in "config.txt", this parameter should be written in days, not in years (maybe it can be used with years too, but I prefer to use it with days)
