I am not responsible for your lost funds, follow further instructions at your own risk

# Binance
Neural Network for trading

#### Notice
Sorry, there is a lot of junk in my code, its my first try to create working neural network.

## Overall
There are some scripts that can automatically buy and sell kryptocurrency based on neural network. All of you need are API key and API secret from your Binance account, how to create it you can find (https://www.binance.com/en/support/faq/how-to-create-api-keys-on-binance-360002502072 "here"). This project already have 1 best individual for traning and one traned model, so you can just run Trading.py with your API settings. More details about each script are written after Getting start

## Getting start
There is two scripts that can't use in the next paragraph, Average_shadows and LoadData. Average shadows calculate average number betwen open price and lowest price and betwen highest price and close price or vice versa.
LoadData loads data :D. And converts it to format that can accept neural network. Before start you need to compile Average Shadows and LoadData
1. Before run any script you need to configure config.txt. In this file you can find parameters for all script from 2 to 5 rows. Settings for genetic algorithm places between 7 and 11 rows e.t.c.
2. Let's start from genetic algorithm. This file finding more adapted individual, after what you can fit them and use for your trading. *WARNING* If you have weak pc, it can take week or more to produce more adapted individual. I rented server with powerfull GPU with 32GB, so you need to check all parameters in config.txt like number of generations or population size. Also if you want to have many deverse individuals, you can increase mutation rate.
3. Now you have the best individual for fit, you need to run TrainBestModel.py to train that model before started. If you see that loss continue fall at the end of fit, you can increase numbers of epochs to get better model.
4. Almost done! You have traned model of neural network that can predict next candle by analysing previous 20 candles. You can run Trading.py to see, how it works and can they predict side of candles. This is gives you the information how profitable model is.
   
