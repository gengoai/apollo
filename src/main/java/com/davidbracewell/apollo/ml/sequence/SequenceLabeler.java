package com.davidbracewell.apollo.ml.sequence;

import com.davidbracewell.apollo.ml.*;
import com.davidbracewell.apollo.ml.preprocess.PreprocessorList;
import lombok.Getter;
import lombok.NonNull;

import java.util.Iterator;

/**
 * The type Sequence labeler.
 *
 * @author David B. Bracewell
 */
public abstract class SequenceLabeler implements Model {
   private static final long serialVersionUID = 1L;
   @Getter
   private final PreprocessorList<Sequence> preprocessors;
   private final TransitionFeatures transitionFeatures;
   private final SequenceValidator validator;
   protected EncoderPair encoderPair;
   private volatile Decoder decoder = new BeamDecoder();

      /**
    * Instantiates a new Model.
    *
    * @param labelEncoder       the label encoder
    * @param featureEncoder     the feature encoder
    * @param preprocessors      the preprocessors
    * @param transitionFeatures the transition features
    * @param validator          the validator
    */
   public SequenceLabeler(LabelEncoder labelEncoder, Encoder featureEncoder, PreprocessorList<Sequence> preprocessors, TransitionFeatures transitionFeatures, SequenceValidator validator) {
      this.encoderPair = new EncoderPair(labelEncoder, featureEncoder);
      this.validator = validator;
      this.preprocessors = preprocessors.getModelProcessors();
      this.transitionFeatures = transitionFeatures;
   }

   /**
    * Estimate double [ ].
    *
    * @param observation the observation
    * @param transitions the transitions
    * @return the double [ ]
    */
   public abstract double[] estimate(Iterator<Feature> observation, Iterator<String> transitions);

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
   public void setDecoder(@NonNull Decoder decoder) {
      this.decoder = decoder;
   }

   @Override
   public EncoderPair getEncoderPair() {
      return encoderPair;
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
    * Gets validator.
    *
    * @return the validator
    */
   public SequenceValidator getValidator() {
      return validator;
   }

   /**
    * Label labeling result.
    *
    * @param sequence the sequence
    * @return the labeling result
    */
   public Labeling label(@NonNull Sequence sequence) {
      return decoder.decode(this, sequence);
   }


}// END OF SequenceLabeler
