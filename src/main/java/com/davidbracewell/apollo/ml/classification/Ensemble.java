package com.davidbracewell.apollo.ml.classification;

import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.ml.EncoderPair;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.preprocess.PreprocessorList;
import com.davidbracewell.collection.counter.Counter;
import com.davidbracewell.collection.counter.HashMapCounter;
import lombok.NonNull;

import java.util.List;

/**
 * @author David B. Bracewell
 */
public class Ensemble extends Classifier {
  private static final long serialVersionUID = 1L;
  List<Classifier> models;

  /**
   * Instantiates a new Classifier.
   *
   * @param encoderPair   the encoder pair
   * @param preprocessors the preprocessors
   */
  protected Ensemble(@NonNull EncoderPair encoderPair, @NonNull PreprocessorList<Instance> preprocessors) {
    super(encoderPair, preprocessors);
  }


  @Override
  public Classification classify(@NonNull Instance instance) {
    Counter<String> results = new HashMapCounter<>();
    for (Classifier model : models) {
      results.merge(model.classify(instance).asCounter());
    }
    results = results.divideBySum();
    double[] dist = new double[numberOfLabels()];
    for (int ci = 0; ci < numberOfLabels(); ci++) {
      dist[ci] = results.get(decodeLabel(ci).toString());
    }
    return new Classification(dist, getLabelEncoder());
  }

  @Override
  public Classification classify(Vector vector) {
    throw new IllegalAccessError();
  }

}// END OF Ensemble
