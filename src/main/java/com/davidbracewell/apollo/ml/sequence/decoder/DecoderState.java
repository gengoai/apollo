package com.davidbracewell.apollo.ml.sequence.decoder;

import java.io.Serializable;

/**
 * <p>Structure used for recording the current decoder state.</p>
 *
 * @author David B. Bracewell
 */
public class DecoderState implements Comparable<DecoderState>, Serializable {
   private static final long serialVersionUID = 1L;
   /**
    * The Sequence probability.
    */
   public final double sequenceProbability;
   /**
    * The State probability.
    */
   public final double stateProbability;
   /**
    * The Tag.
    */
   public final String tag;
   /**
    * The Prev.
    */
   public final DecoderState previousState;
   /**
    * The Index.
    */
   public final int index;

   /**
    * Instantiates a new Decoder state.
    *
    * @param stateProbability the state probability
    * @param tag              the tag
    */
   public DecoderState(double stateProbability, String tag) {
      this(null, stateProbability, tag);
   }

   /**
    * Instantiates a new Decoder state.
    *
    * @param previousState    the previous state
    * @param stateProbability the state probability
    * @param tag              the tag
    */
   public DecoderState(DecoderState previousState, double stateProbability, String tag) {
      this.index = previousState == null ? 0 : previousState.index + 1;
      this.stateProbability = stateProbability;
      if (previousState == null) {
         this.sequenceProbability = stateProbability;//Math.log(stateProbability);
      } else {
         this.sequenceProbability = previousState.sequenceProbability + stateProbability;//Math.log(stateProbability);
      }
      this.tag = tag;
      this.previousState = previousState;
   }

   @Override
   public int compareTo(DecoderState o) {
      return Double.compare(this.sequenceProbability, o.sequenceProbability);
   }

}// END OF DecoderState
