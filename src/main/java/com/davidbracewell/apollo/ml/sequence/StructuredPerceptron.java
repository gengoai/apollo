package com.davidbracewell.apollo.ml.sequence;

import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.ml.Encoder;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.classification.ClassifierResult;
import com.davidbracewell.apollo.ml.preprocess.PreprocessorList;
import lombok.NonNull;

import java.util.Iterator;

/**
 * @author David B. Bracewell
 */
public class StructuredPerceptron extends SequenceLabeler {
  Vector[] weights;

  /**
   * Instantiates a new Model.
   *
   * @param labelEncoder   the label encoder
   * @param featureEncoder the feature encoder
   * @param preprocessors  the preprocessors
   */
  public StructuredPerceptron(@NonNull Encoder labelEncoder, @NonNull Encoder featureEncoder, PreprocessorList<Sequence> preprocessors, TransitionFeatures transitionFeatures) {
    super(labelEncoder, featureEncoder, preprocessors, transitionFeatures);
  }

  @Override
  public ClassifierResult estimateInstance(Instance instance) {
    Vector v = instance.toVector(getFeatureEncoder());
    double[] distribution = new double[numberOfLabels()];
    double max = 0;
    for (Iterator<Vector.Entry> iterator = v.nonZeroIterator(); iterator.hasNext(); ) {
      Vector.Entry de = iterator.next();
      for (int ci = 0; ci < numberOfLabels(); ci++) {
        distribution[ci] = weights[ci].get(de.index) * de.value;
        max = Math.max(max, distribution[ci]);
      }
    }

    for (int ci = 0; ci < numberOfLabels(); ci++) {
      distribution[ci] = distribution[ci] / max;
    }
    return new ClassifierResult(distribution, getLabelEncoder());
  }

}// END OF StructuredPerceptron
