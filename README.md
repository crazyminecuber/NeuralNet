# Neural network from scratch

This is a neural network that trains to recognice handwritten digits from the famous MNIST dataset.

I followed the exelent book at http://neuralnetworksanddeeplearning.com/ , but implemented it in C++ instead of python to make sure that I had to understand it to get it working.

For the linear algebra i used the eigen library (https://eigen.tuxfamily.org/index.php?title=Main_Page)

## Running

```
make
make run
```

This starts the training process and you can see it learn in real time by watching the percentage of correct guesses go up.
