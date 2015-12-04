package com.davidbracewell.apollo.ml.classification.rule;

import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.ml.Encoder;
import com.davidbracewell.apollo.ml.classification.Classifier;
import com.davidbracewell.apollo.ml.classification.ClassifierResult;
import com.davidbracewell.apollo.ml.preprocess.PreprocessorList;
import com.davidbracewell.collection.Counter;
import com.davidbracewell.collection.Counters;

/**
 * @author David B. Bracewell
 */
public class ZeroR extends Classifier {

  final Counter<String> distribution = Counters.newHashMapCounter();

  /**
   * Instantiates a new Classifier.
   *
   * @param labelEncoder   the label encoder
   * @param featureEncoder the feature encoder
   */
  protected ZeroR(Encoder labelEncoder, Encoder featureEncoder) {
    super(labelEncoder, featureEncoder, PreprocessorList.empty());
  }

  @Override
  public ClassifierResult classify(Vector vector) {
    return new ClassifierResult(Counters.newHashMapCounter(distribution));
  }

}// END OF ZeroR
