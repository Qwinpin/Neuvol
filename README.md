# ![logo](logo.png)Neuvol

*Genetic algorithm to find the best neural network architecture with Keras*

Neuvol is a genetic algorithm API for generating neural networks based on Keras. The main idea is to work with data only, without direct architecture constructing.

##### Grow your own neural network!

- Data in -> Neural Network Architecture out
- A large number of allowed layers types
- Complex crossing approaches
- Modular structure
- Keep the whole mutation history, resolve "dead" mutations such as cycling connections
- All possible architecture manipulations: add/remove layer/connection
- All possible structures: branches, skip-connections, combination of layers with different dimensions
- On the fly shape analysis: add reshape for concatenations or for connecting layer with 3 output dimensions and layer with 1 input (like dense after convolution)

### Features:

- Supported data types: texts, images
- The list of supported layers is constantly expanding and contains most popular of them

### TODO

- [x] Architectures distribution generation
- [x] Images support
- [x] More available layers
- [x] Logo
- [x] Serialiser
- [x] Complex layer generation
- [ ] Experimental study
- [ ] Visualization
- [ ] Pytorch support
