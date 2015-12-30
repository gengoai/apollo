package com.davidbracewell.apollo.ml.classification.neural;

import com.davidbracewell.apollo.DifferentiableFunction;
import com.davidbracewell.apollo.linalg.DenseMatrix;
import com.davidbracewell.apollo.linalg.Matrix;
import com.davidbracewell.apollo.linalg.SparseVector;
import com.davidbracewell.apollo.linalg.Vector;

import java.io.Serializable;

/**
 * @author David B. Bracewell
 */
public class Layer implements Serializable {
  private static final long serialVersionUID = 1L;
  private final Neuron[] neurons;
  private volatile Matrix matrix;
  private int inputSize;

  public Layer(int inputSize, int outputSize, DifferentiableFunction activationFunction) {
    matrix = DenseMatrix.random(outputSize, inputSize);
    this.inputSize = inputSize;
    this.neurons = new Neuron[outputSize];
    for (int i = 0; i < outputSize; i++) {
      this.neurons[i] = new Neuron(inputSize, activationFunction);
    }
  }

  public int inputSize() {
    return inputSize;
  }

  public int outputSize() {
    return neurons.length;
  }

  public Vector multiply(Vector delta) {
    Vector out = new SparseVector(outputSize());
    for (int i = 0; i < neurons.length; i++) {
      out.set(i, neurons[i].dot(delta));
    }
    return out;
  }

  public Vector evaluate(Vector input) {
    return matrix.scale(input).mapSelf(new Sigmoid()).row(0);
//    System.out.println(matrix.numberOfRows() + " : " + matrix.numberOfColumns() + " : " + input.dimension());
//    System.out.println(matrix.scale(input));
//    Vector output = new SparseVector(neurons.length);
//    for (int i = 0; i < neurons.length; i++) {
//      output.set(i, neurons[i].activate(input));
//    }
//    return output;
  }

  public Vector gradient(Vector input) {
    Vector gradient = new SparseVector(neurons.length);
    for (int i = 0; i < neurons.length; i++) {
      gradient.set(i, neurons[i].gradient(input));
    }
    return gradient;
  }

  public void update(Vector delta) {
    for (int i = 0; i < neurons.length; i++) {
      neurons[i].update(delta);
    }
  }


}// END OF Layer
