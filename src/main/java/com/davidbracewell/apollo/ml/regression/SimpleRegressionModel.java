package com.davidbracewell.apollo.ml.regression;

import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.ml.EncoderPair;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.preprocess.PreprocessorList;
import com.davidbracewell.collection.counter.Counter;
import com.davidbracewell.collection.counter.Counters;
import lombok.NonNull;

/**
 * @author David B. Bracewell
 */
public class SimpleRegressionModel extends Regression {
  private static final long serialVersionUID = 1L;
  Vector weights;
  double bias;

  /**
   * Instantiates a new Regression.
   *
   * @param encoderPair   the encoder pair
   * @param preprocessors the preprocessors
   */
  public SimpleRegressionModel(@NonNull EncoderPair encoderPair, @NonNull PreprocessorList<Instance> preprocessors) {
    super(encoderPair, preprocessors);
  }

  @Override
  public double estimate(@NonNull Vector vector) {
    return bias + weights.dot(vector);
  }

  @Override
  public Counter<String> getFeatureWeights() {
    Counter<String> out = Counters.newCounter();
    out.set("***BIAS***", bias);
    weights.forEachSparse(e -> out.set(decodeFeature(e.index).toString(), e.value));
    return out;
  }

}// END OF SimpleRegressionModel
