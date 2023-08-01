!I am not responsible for your lost funds, follow further instructions at your own risk!

# Binance
Neural Network for trading

#### Notice
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
   
