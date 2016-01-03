package com.davidbracewell.apollo.ml.classification.neural;

import com.davidbracewell.apollo.linalg.DenseMatrix;
import com.davidbracewell.apollo.linalg.Matrix;

import java.io.Serializable;

/**
 * @author David B. Bracewell
 */
public class Layer implements Serializable {
  private static final long serialVersionUID = 1L;
  private Activation activation;
  private volatile Matrix matrix;
  private double bias = 1.0;

  public Layer(int inputSize, int outputSize, Activation activation) {
    this.matrix = DenseMatrix.random(inputSize, outputSize, -0.5, 0.5);
    this.activation = activation;
  }

  public Matrix getMatrix() {
    return matrix;
  }

  public Matrix evaluate(Matrix input) {
    return input.multiply(matrix).incrementSelf(bias).mapSelf(activation);
  }

  public Matrix gradient(Matrix input) {
    return input.map(activation::gradientOfResult);
  }

}// END OF Layer
