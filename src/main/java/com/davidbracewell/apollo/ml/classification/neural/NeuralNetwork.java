package com.davidbracewell.apollo.ml.classification.neural;

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
  Network network;

  /**
   * Instantiates a new Classifier.
   *
   * @param encoderPair   the encoder pair
   * @param preprocessors the preprocessors
   */
  protected NeuralNetwork(@NonNull EncoderPair encoderPair, @NonNull PreprocessorList<Instance> preprocessors) {
    super(encoderPair, preprocessors);
  }

  @Override
  public Classification classify(Vector vector) {
    return network.evaluate(vector);
  }


}// END OF NeuralNetwork
