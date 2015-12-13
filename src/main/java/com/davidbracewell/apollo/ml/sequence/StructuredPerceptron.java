package com.davidbracewell.apollo.ml.sequence;

import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.ml.Encoder;
import com.davidbracewell.apollo.ml.Feature;
import com.davidbracewell.apollo.ml.preprocess.PreprocessorList;
import com.davidbracewell.collection.LRUMap;
import com.google.common.base.Stopwatch;
import com.google.common.primitives.Floats;
import lombok.NonNull;

import java.util.Iterator;
import java.util.Map;

/**
 * @author David B. Bracewell
 */
public class StructuredPerceptron extends SequenceLabeler {
  final int numberOfClasses;
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
    this.numberOfClasses = labelEncoder.size();
  }

  @Override
  public double[] estimate(Iterator<Feature> observation, Iterator<String> transitions) {
    double[] distribution = new double[numberOfClasses];
    while (observation.hasNext()) {
      Feature feature = observation.next();
      int index = (int) getFeatureEncoder().encode(feature.getName());
      if (index != -1) {
        for (int ci = 0; ci < numberOfClasses; ci++) {
          distribution[ci] += weights[ci].get(index);
        }
      }
    }

    while (transitions.hasNext()) {
      int index = (int) getFeatureEncoder().encode(transitions.next());
      if (index != -1) {
        for (int ci = 0; ci < numberOfClasses; ci++) {
          distribution[ci] += weights[ci].get(index);
        }
      }
    }

    return distribution;
  }

}// END OF StructuredPerceptron
