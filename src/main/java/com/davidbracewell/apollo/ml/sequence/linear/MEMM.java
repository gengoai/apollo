package com.davidbracewell.apollo.ml.sequence.linear;

import com.davidbracewell.apollo.ml.Encoder;
import com.davidbracewell.apollo.ml.Feature;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.classification.linear.LibLinearModel;
import com.davidbracewell.apollo.ml.preprocess.PreprocessorList;
import com.davidbracewell.apollo.ml.sequence.Sequence;
import com.davidbracewell.apollo.ml.sequence.SequenceLabeler;
import com.davidbracewell.apollo.ml.sequence.SequenceValidator;
import com.davidbracewell.apollo.ml.sequence.TransitionFeatures;
import com.google.common.collect.Lists;

import java.util.Iterator;
import java.util.List;

/**
 * @author David B. Bracewell
 */
public class MEMM extends SequenceLabeler {
  private static final long serialVersionUID = 1L;
  LibLinearModel model;

  /**
   * Instantiates a new Model.
   *
   * @param labelEncoder       the label encoder
   * @param featureEncoder     the feature encoder
   * @param preprocessors      the preprocessors
   * @param transitionFeatures the transition features
   * @param validator          the Validator
   */
  public MEMM(Encoder labelEncoder, Encoder featureEncoder, PreprocessorList<Sequence> preprocessors, TransitionFeatures transitionFeatures, SequenceValidator validator) {
    super(labelEncoder, featureEncoder, preprocessors, transitionFeatures, validator);
  }

  @Override
  public double[] estimate(Iterator<Feature> observation, Iterator<String> transitions) {
    List<Feature> features = Lists.newArrayList(observation);
    while (transitions.hasNext()) {
      features.add(Feature.TRUE(transitions.next()));
    }
    return model.classify(Instance.create(features).toVector(getEncoderPair())).distribution();
  }

}// END OF MEMM
