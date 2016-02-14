package com.davidbracewell.apollo.ml.sequence.linear;

import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.ml.Encoder;
import com.davidbracewell.apollo.ml.Feature;
import com.davidbracewell.apollo.ml.preprocess.PreprocessorList;
import com.davidbracewell.apollo.ml.sequence.Sequence;
import com.davidbracewell.apollo.ml.sequence.SequenceLabeler;
import com.davidbracewell.apollo.ml.sequence.SequenceValidator;
import com.davidbracewell.apollo.ml.sequence.TransitionFeatures;
import lombok.NonNull;

import java.util.Iterator;

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
  public StructuredPerceptron(@NonNull Encoder labelEncoder, @NonNull Encoder featureEncoder, @NonNull PreprocessorList<Sequence> preprocessors, @NonNull TransitionFeatures transitionFeatures, @NonNull SequenceValidator validator) {
    super(labelEncoder, featureEncoder, preprocessors, transitionFeatures, validator);
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
