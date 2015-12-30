package com.davidbracewell.apollo.ml.classification.neural;

import com.davidbracewell.apollo.linalg.DenseMatrix;
import com.davidbracewell.apollo.linalg.Matrix;
import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.ml.EncoderPair;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.classification.Classification;
import com.davidbracewell.apollo.ml.classification.Classifier;
import com.davidbracewell.apollo.ml.preprocess.PreprocessorList;
import lombok.NonNull;

/**
 * @author David B. Bracewell
 */
public class NeuralNetwork extends Classifier {
  private static final long serialVersionUID = 1L;

  Layer[] layers;

  /**
   * Instantiates a new Classifier.
   *
   * @param encoderPair   the encoder pair
   * @param preprocessors the preprocessors
   */
  protected NeuralNetwork(@NonNull EncoderPair encoderPair, @NonNull PreprocessorList<Instance> preprocessors) {
    super(encoderPair, preprocessors);
  }

  public static void main(String[] args) throws Exception {
    Matrix X = new DenseMatrix(new double[][]{
      new double[]{0, 0, 1},
      new double[]{0, 1, 1},
      new double[]{1, 0, 1},
      new double[]{1, 1, 1},
      new double[]{1, 1, 1},
    });

    Matrix XTranspose = X.transpose();

    Matrix Y = new DenseMatrix(new double[][]{
      new double[]{0, 1, 1, 0, 1}
    }).transpose();

    Matrix syn0 = DenseMatrix.random(3, 4).mapSelf(d -> 2.0 * d - 1);
    Matrix syn1 = DenseMatrix.random(4, 1).mapSelf(d -> 2.0 * d - 1);

    Sigmoid sigmoid = new Sigmoid();
    for (int j = 0; j < 60000; j++) {
      Matrix l1 = X.multiply(syn0).mapSelf(sigmoid);
      Matrix l2 = l1.multiply(syn1).mapSelf(sigmoid);

      Matrix l2Gradient = l2.map(sigmoid::gradientOfResult);
      Matrix l2_delta = Y.subtract(l2).scaleSelf(l2Gradient);


      Matrix l1Gradient = l1.map(sigmoid::gradientOfResult);
      Matrix l1_delta = l2_delta.multiply(syn1.transpose()).scaleSelf(l1Gradient);

      syn1.addSelf(l1.transpose().multiply(l2_delta));
      syn0.addSelf(XTranspose.multiply(l1_delta));
    }

    Matrix l2 = X.multiply(syn0)
      .mapSelf(sigmoid::applyAsDouble)
      .multiply(syn1)
      .mapSelf(sigmoid::applyAsDouble);
    System.out.println(l2);

  }

  @Override
  public Classification classify(Vector vector) {
    Matrix v = vector.toMatrix();
    for (Layer layer : layers) {
      v = layer.evaluate(v);
    }
    return new Classification(v.row(0).toArray(), getLabelEncoder());
  }


}// END OF NeuralNetwork
