# Epicenter Computation
This directory presents experiments on tools to compute the epicenter of a wave propagating from various measures.
It contains different notebooks:
- `GenerateDatasets.ipynb`: Generate training and testing datasets
- `Regressions.ipynb`: Examples of statistical regression tools when the wave amplitude is known at all the points of the studied field but only at given time steps
- [Interrogator](Interrogator/): Regressions techniques applied on the case where the wave amplitude is continuously known at specific spatial points
	- `RegressionInterrogator.ipynb`: Statistical regression tools
	- `FeedForwardNeuralNetwork.ipynb`: A simple 2-layers feed-forward neural network analysis
- [Results](/Results): Plots of the different experimental results

