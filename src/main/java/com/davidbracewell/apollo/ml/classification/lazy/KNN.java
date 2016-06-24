package com.davidbracewell.apollo.ml.classification.lazy;

import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.linalg.VectorStore;
import com.davidbracewell.apollo.ml.EncoderPair;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.classification.Classification;
import com.davidbracewell.apollo.ml.classification.Classifier;
import com.davidbracewell.apollo.ml.preprocess.PreprocessorList;
import lombok.NonNull;

/**
 * @author David B. Bracewell
 */
public class KNN extends Classifier {
  VectorStore<Integer> vectors;
  int K;

  /**
   * Instantiates a new Classifier.
   *
   * @param encoderPair   the encoder pair
   * @param preprocessors the preprocessors
   */
  protected KNN(@NonNull EncoderPair encoderPair, @NonNull PreprocessorList<Instance> preprocessors) {
    super(encoderPair, preprocessors);
  }

  @Override
  public Classification classify(Vector vector) {
    double[] distribution = new double[numberOfLabels()];
    vectors.nearest(vector, K).forEach(slv ->
      distribution[(int) slv.getLabel()] += slv.getScore()
    );
    return new Classification(distribution, getLabelEncoder());
  }

}// END OF KNN
