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

  @Override
  public Classification classify(Vector vector) {
    Vector v = vector;
    for (Layer layer : layers) {
      v = layer.evaluate(v);
    }
    return new Classification(v.toArray(), getLabelEncoder());
  }


}// END OF NeuralNetwork
