package com.davidbracewell.apollo.ml.classification.neural;

import com.davidbracewell.apollo.DifferentiableFunction;
import com.davidbracewell.apollo.linalg.SparseVector;
import com.davidbracewell.apollo.linalg.Vector;

import java.io.Serializable;

/**
 * @author David B. Bracewell
 */
public class Neuron implements Serializable {
  private static final long serialVersionUID = 1L;
  private final DifferentiableFunction activation;
  private Vector weights;

  public Neuron(int size, DifferentiableFunction activation) {
    this.activation = activation;
    this.weights = SparseVector.randomGaussian(size);
  }

  public double dot(Vector input) {
    return weights.dot(input);
  }

  public double activate(Vector input) {
    return activation.applyAsDouble(input.dot(weights));
  }

  public void update(Vector vector) {
    weights.addSelf(vector);
  }

  public double gradient(Vector input) {
    return activation.gradient(input.dot(weights));
  }

}// END OF Neuron
