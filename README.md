# Unsupervised Learning
A set of unsupervised learning models implementations. These models include:
* Self organizing map (or Kohonen Network)
* A linear perceptron using Oja's rule
* Hopfield network

## Dependencies
To be able to run the different models, be sure to have installed the following python dependencies:
* numpy
* pandas
* matplotlib
* seaborn
* scipy
* sklearn

## Running the examples
* Self Organizing Map example: `python kohonen.py`, it will generate clusters based on the information in the file europe.csv, and display a heatmap with the clusters, and a matrix representing the average distance between each neuron and it's neighbors.
* Linear perceptron with Oja's rule example: `python oja.py`, it will print the weights representing the principal component for the dataset europe.csv
* Hopfield example: `python hopfield.py <letter> <number>`. `<letter>` must be a letter belonging to the set {S, B, L, A}. `<number>` is the amount of noise to add to `<letter>`. Each letter is a vector of 25 elements. When executed, noise will be added to the given letter, and the resultant pattern with noise will be predicted by the model. The output will indicate the sequence of steps the model took to arrive to the predicted pattern. Concrete example usage: `python hopfield.py S 10` will add noise to 10 elements of the letter S, and then predict the resultant pattern with noise.
