package com.davidbracewell.apollo.ml.classification.neural;

import com.davidbracewell.apollo.linalg.Matrix;

/**
 * @author David B. Bracewell
 */
public class SoftmaxLayer extends Layer {
  private static final long serialVersionUID = 1L;

  public SoftmaxLayer(int inputSize, int outputSize) {
    super(inputSize, outputSize, new Sigmoid());
  }

  @Override
  public Matrix evaluate(Matrix input) {
    Matrix m = super.evaluate(input);
    double max = m.row(0).max();
    m.incrementSelf(-max);
    m.mapSelf(Math::exp);
    double sum = m.row(0).sum();
    return m.scaleSelf(1d / sum);
  }

}// END OF SoftmaxLayer
