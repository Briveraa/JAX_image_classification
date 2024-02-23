# Image Classification with a Convolutional Neural Network
![](https://flax.readthedocs.io/en/latest/_static/flax.png)
---
### Description 
My newest project is an experiment in image classification using a relatively small dataset. This project uses a Convolutional Neural Network with the Flax library to classify images of horses and humans. 

Image classification is a fundamental task in computer vision, with applications ranging from medical diagnostics to autonomous driving. By building this image classifier, using frameworks such as JAX and Flax, this project serves as a stepping stone towards tackling more complex computer vision tasks. 


### Data
This project utilized TensorFlow Datasets (TFDS) to access the "Horses vs Humans" dataset, comprising 1,027 training and 257 validation images. Two key aspects of this dataset are worth noting. Firstly, its small size can lead to instability during model training. Secondly, a notable distinction exists between the backgrounds of training and test images, with multicolored backgrounds in the former and white backgrounds in the latter. This difference will impact model performance unless the train and test sets are combined, followed by re-splitting for balanced training and evaluation.


### Model
This model is defined using the Flax Linen API which is built on top of JAX. You can define a neural network using Flax in two ways: explicitly using the setup method or in-line using the @compact decorator. According to Flax documentation, "Both of these approaches are perfectly valid, behave the same way, and interoperate with all of Flax ." I chose to use the setup method for this project. 

```python
class CNN(nn.Module):
  """ A simple CNN model using setup method """
  hidden_dim: int = 16 

  def setup(self):
    self.conv1 = nn.Conv(features=self.hidden_dim, kernel_size=(3,3))
    self.pool1 = nn.max_pool
    self.conv2 = nn.Conv(features=self.hidden_dim *2, kernel_size=(3,3))
    self.pool2 = nn.max_pool
    self.dense1 = nn.Dense(features= 64, kernel_init=nn.initializers.he_normal())
    self.dense2 = nn.Dense(features=2)

  #forward pass
  def __call__(self, x):
    x = self.conv1(x)
    x = nn.relu(x)
    x = self.pool1(x, window_shape=(2,2))
    x = self.conv2(x)
    x = nn.relu(x)
    x = self.pool2(x, window_shape=(2,2))
    x = jnp.reshape(x, (x.shape[0], -1)) 
    x = self.dense1(x)
    x = nn.relu(x)
    return self.dense2(x)
```

### Conclusion
This project serves as an exploration into machine learning model development for image classification using JAX. Despite achieving 86% accuracy on the test set, there is room for improvement. Addressing the dataset's limitations by combining and re-splitting the data, as discussed earlier, could enhance performance. Additionally, augmenting the dataset with image transformations could increase its size and diversity.

This project gave me valuable insights into JAX, neural network architectures, and optimization techniques. Ultimately, this project lays the foundation for further exploration and refinement of image classification models.
