# neural_network_design.py

"""
Neural Network Design for Energy-Efficient Neural Networks

This module contains functions for designing energy-efficient neural networks with a focus on lightweight architectures and pruning techniques.

Techniques Used:
- Model pruning
- Quantization
- Efficient architectures

Libraries/Tools:
- tensorflow
- keras

"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow_model_optimization.sparsity import keras as sparsity
import os

class NeuralNetworkDesign:
    def __init__(self, input_shape=(224, 224, 3), num_classes=10):
        """
        Initialize the NeuralNetworkDesign class.
        
        :param input_shape: tuple, shape of the input data
        :param num_classes: int, number of output classes
        """
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build_model(self):
        """
        Build a basic CNN model.
        
        :return: Model, compiled Keras model
        """
        inputs = Input(shape=self.input_shape)
        x = Conv2D(32, (3, 3), activation='relu')(inputs)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(128, (3, 3), activation='relu')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)

        model = Model(inputs, outputs)
        model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def prune_model(self, model):
        """
        Apply pruning to the model.
        
        :param model: Model, Keras model to be pruned
        :return: Model, pruned Keras model
        """
        pruning_params = {
            'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=0.0, final_sparsity=0.5, begin_step=0, end_step=2000)
        }
        pruned_model = sparsity.prune_low_magnitude(model, **pruning_params)
        pruned_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
        return pruned_model

    def quantize_model(self, model, representative_data_gen):
        """
        Apply quantization to the model.
        
        :param model: Model, Keras model to be quantized
        :param representative_data_gen: generator, generator for representative dataset
        :return: Model, quantized TFLite model
        """
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_data_gen
        quantized_model = converter.convert()
        return quantized_model

    def save_model(self, model, model_path='models/optimized_model.h5'):
        """
        Save the Keras model to a file.
        
        :param model: Model, Keras model to be saved
        :param model_path: str, path to save the model
        """
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save(model_path)
        print(f"Model saved to {model_path}")

    def save_quantized_model(self, quantized_model, model_path='models/quantized_model.tflite'):
        """
        Save the quantized TFLite model to a file.
        
        :param quantized_model: Model, quantized TFLite model to be saved
        :param model_path: str, path to save the quantized model
        """
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, 'wb') as f:
            f.write(quantized_model)
        print(f"Quantized model saved to {model_path}")

def representative_data_gen():
    # This function should yield batches of data to be used for representative dataset
    # for quantization. Here we use dummy data for demonstration.
    for _ in range(100):
        yield [np.random.rand(1, 224, 224, 3).astype(np.float32)]

if __name__ == "__main__":
    input_shape = (224, 224, 3)
    num_classes = 10

    nn_design = NeuralNetworkDesign(input_shape=input_shape, num_classes=num_classes)
    
    # Build the model
    model = nn_design.build_model()
    print("Model built.")
    
    # Apply pruning
    pruned_model = nn_design.prune_model(model)
    print("Model pruned.")
    
    # Save the pruned model
    nn_design.save_model(pruned_model, model_path='models/pruned_model.h5')
    
    # Apply quantization
    quantized_model = nn_design.quantize_model(pruned_model, representative_data_gen)
    print("Model quantized.")
    
    # Save the quantized model
    nn_design.save_quantized_model(quantized_model, model_path='models/quantized_model.tflite')
