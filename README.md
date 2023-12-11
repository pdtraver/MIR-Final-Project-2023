There are a few main project files in this repo.
I have two models, one from Kaggle in the 'notebook416bf8ddb8.ipynb' file, 
which was copied into the 'kaggle_playground.ipynb' file for me to tweak 
parameters while providing reference to the original model. The Kaggle model is a VAE
used for audio generation over the gtzan dataset. The second model
lives in vae.ipynb as is a model across the MNIST dataset, though 'vae_playground.ipynb' attempts
to use this construction for audio generation. The second model, however, is used primiarily for its
plotting functions, meaning the kaggle_playground.ipynb file is where the bulk of the work is done.

Within this file you'll see the use of the Frechet Audio Distance (FAD) score, Inception Score (IS) and Precision/Recall scores
for the model.
