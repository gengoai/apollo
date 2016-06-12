package com.davidbracewell.apollo.ml.classification.rule;

import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.ml.EncoderPair;
import com.davidbracewell.apollo.ml.classification.Classification;
import com.davidbracewell.apollo.ml.classification.Classifier;
import com.davidbracewell.apollo.ml.preprocess.PreprocessorList;
import com.davidbracewell.collection.HashMapMultiCounter;
import com.davidbracewell.collection.MultiCounter;
import lombok.NonNull;

/**
 * The type Zero r.
 *
 * @author David B. Bracewell
 */
public class ZeroR extends Classifier {

  /**
   * The Distribution.
   */
  double[] distribution;

  /**
   * Instantiates a new Classifier.
   *
   * @param encoderPair the encoder pair
   */
  protected ZeroR(@NonNull EncoderPair encoderPair) {
    super(encoderPair, PreprocessorList.empty());
  }

  @Override
  public Classification classify(Vector vector) {
    return new Classification(distribution, getLabelEncoder());
  }

  @Override
  public MultiCounter<String, String> getModelParameters() {
    MultiCounter<String, String> weights = new HashMapMultiCounter<>();
    for (int i = 0; i < distribution.length; i++) {
      weights.set("*", getLabelEncoder().decode(i).toString(), distribution[i]);
    }
    return weights;
  }

}// END OF ZeroR
