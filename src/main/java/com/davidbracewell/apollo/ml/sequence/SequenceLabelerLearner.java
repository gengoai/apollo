package com.davidbracewell.apollo.ml.sequence;

import com.davidbracewell.apollo.ml.Learner;

/**
 * @author David B. Bracewell
 */
public abstract class SequenceLabelerLearner extends Learner<Sequence, SequenceLabeler> {
  private static final long serialVersionUID = 1L;
  protected Decoder decoder;
  protected TransitionFeatures transitionFeatures = TransitionFeatures.FIRST_ORDER;

  public Decoder getDecoder() {
    return decoder;
  }

  public void setDecoder(Decoder decoder) {
    this.decoder = decoder;
  }

  public TransitionFeatures getTransitionFeatures() {
    return transitionFeatures;
  }

  public void setTransitionFeatures(TransitionFeatures transitionFeatures) {
    this.transitionFeatures = transitionFeatures;
  }
}// END OF SequenceLabelerLearner
