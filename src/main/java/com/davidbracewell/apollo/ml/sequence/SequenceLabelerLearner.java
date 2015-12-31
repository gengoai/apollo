package com.davidbracewell.apollo.ml.sequence;

import com.davidbracewell.apollo.ml.Learner;
import com.davidbracewell.apollo.ml.sequence.decoder.BeamDecoder;
import com.davidbracewell.apollo.ml.sequence.decoder.Decoder;
import lombok.NonNull;

/**
 * @author David B. Bracewell
 */
public abstract class SequenceLabelerLearner extends Learner<Sequence, SequenceLabeler> {
  private static final long serialVersionUID = 1L;
  protected Decoder decoder = new BeamDecoder(5);
  protected TransitionFeatures transitionFeatures = TransitionFeatures.FIRST_ORDER;
  protected SequenceValidator validator = SequenceValidator.ALWAYS_TRUE;

  public Decoder getDecoder() {
    return decoder;
  }

  public void setDecoder(Decoder decoder) {
    this.decoder = decoder;
  }

  public TransitionFeatures getTransitionFeatures() {
    return transitionFeatures;
  }

  public void setTransitionFeatures(@NonNull TransitionFeatures transitionFeatures) {
    this.transitionFeatures = transitionFeatures;
  }

  public SequenceValidator getValidator() {
    return validator;
  }

  public void setValidator(@NonNull SequenceValidator validator) {
    this.validator = validator;
  }
}// END OF SequenceLabelerLearner
