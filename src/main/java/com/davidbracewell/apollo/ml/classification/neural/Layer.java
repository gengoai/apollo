package com.davidbracewell.apollo.ml.classification.neural;

import com.davidbracewell.apollo.DifferentiableFunction;
import com.davidbracewell.apollo.linalg.SparseVector;
import com.davidbracewell.apollo.linalg.Vector;

import java.io.Serializable;

/**
 * @author David B. Bracewell
 */
public class Layer implements Serializable {
  private static final long serialVersionUID = 1L;
  private final Neuron[] neurons;
  private int inputSize;

  public Layer(int inputSize, int outputSize, DifferentiableFunction activationFunction) {
    this.inputSize = inputSize;
    this.neurons = new Neuron[outputSize];
    for (int i = 0; i < outputSize; i++) {
      this.neurons[i] = new Neuron(inputSize, activationFunction);
    }
  }

  public Vector evaluate(Vector input) {
    Vector output = new SparseVector(neurons.length);
    for (int i = 0; i < neurons.length; i++) {
      output.set(i, neurons[i].activate(input));
    }
    return output;
  }

  public Vector gradient(Vector input) {
    Vector gradient = new SparseVector(inputSize);
    for (int i = 0; i < neurons.length; i++) {
      gradient.set(i, neurons[i].gradient(input));
    }
    return gradient;
  }


}// END OF Layer
