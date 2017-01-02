package com.davidbracewell.apollo.ml.sequence;

import com.davidbracewell.apollo.ml.Learner;
import com.davidbracewell.apollo.ml.data.Dataset;
import lombok.NonNull;

/**
 * The type Sequence labeler learner.
 *
 * @author David B. Bracewell
 */
public abstract class SequenceLabelerLearner extends Learner<Sequence, SequenceLabeler> {
   private static final long serialVersionUID = 1L;
   /**
    * The Decoder.
    */
   protected Decoder decoder = new BeamDecoder(5);
   /**
    * The Transition features.
    */
   protected TransitionFeatures transitionFeatures = TransitionFeatures.FIRST_ORDER;
   /**
    * The Validator.
    */
   protected SequenceValidator validator = SequenceValidator.ALWAYS_TRUE;

   /**
    * Gets decoder.
    *
    * @return the decoder
    */
   public Decoder getDecoder() {
      return decoder;
   }

   /**
    * Sets decoder.
    *
    * @param decoder the decoder
    */
   public void setDecoder(Decoder decoder) {
      this.decoder = decoder;
   }

   /**
    * Gets transition features.
    *
    * @return the transition features
    */
   public TransitionFeatures getTransitionFeatures() {
      return transitionFeatures;
   }

   /**
    * Sets transition features.
    *
    * @param transitionFeatures the transition features
    */
   public void setTransitionFeatures(@NonNull TransitionFeatures transitionFeatures) {
      this.transitionFeatures = transitionFeatures;
   }

   @Override
   public SequenceLabeler train(@NonNull Dataset<Sequence> dataset) {
      dataset.encode();
      transitionFeatures.fitTransitionsFeatures(dataset);
      dataset.getEncoderPair().freeze();
      SequenceLabeler model = trainImpl(dataset);
      model.finishTraining();
      model.getFeatureEncoder().freeze();
      return model;
   }


   /**
    * Gets validator.
    *
    * @return the validator
    */
   public SequenceValidator getValidator() {
      return validator;
   }

   /**
    * Sets validator.
    *
    * @param validator the validator
    */
   public void setValidator(@NonNull SequenceValidator validator) {
      this.validator = validator;
   }
}// END OF SequenceLabelerLearner
