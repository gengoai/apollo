package com.davidbracewell.apollo.ml.classification.neural;

import com.davidbracewell.apollo.function.DifferentiableFunction;
import com.davidbracewell.apollo.linalg.SparseVector;
import com.davidbracewell.apollo.linalg.Vector;

import java.io.Serializable;

/**
 * @author David B. Bracewell
 */
public class Neuron implements Serializable {
  private final DifferentiableFunction activation;
  private Vector weights;

  public Neuron(int size, DifferentiableFunction activation) {
    this.activation = activation;
    this.weights = SparseVector.randomGaussian(size);
  }

  public double activate(Vector input) {
    return activation.applyAsDouble(input.dot(weights));
  }

  public double gradient(Vector input) {
    return activation.gradientAsDouble(input.dot(weights));
  }

}// END OF Neuron
