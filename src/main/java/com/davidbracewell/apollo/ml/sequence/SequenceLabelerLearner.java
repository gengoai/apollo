package com.davidbracewell.apollo.ml.sequence;

import com.davidbracewell.apollo.ml.Learner;

/**
 * @author David B. Bracewell
 */
public abstract class SequenceLabelerLearner extends Learner<Sequence, SequenceLabeler> {
  private static final long serialVersionUID = 1L;
  public Decoder decoder;


  public Decoder getDecoder() {
    return decoder;
  }

  public void setDecoder(Decoder decoder) {
    this.decoder = decoder;
  }
}// END OF SequenceLabelerLearner
