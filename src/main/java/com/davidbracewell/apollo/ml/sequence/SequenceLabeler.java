package com.davidbracewell.apollo.ml.sequence;

import com.davidbracewell.apollo.ml.Feature;
import com.davidbracewell.apollo.ml.Model;
import com.davidbracewell.apollo.ml.encoder.EncoderPair;
import com.davidbracewell.apollo.ml.preprocess.PreprocessorList;
import lombok.Getter;
import lombok.NonNull;
import lombok.Setter;

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
   @Getter
   private final TransitionFeature transitionFeatures;
   @Getter
   private final SequenceValidator validator;
   protected EncoderPair encoderPair;
   @Getter
   @Setter
   private volatile Decoder decoder = new BeamDecoder();

   /**
    * Instantiates a new Model.
    *
    * @param learner the learner
    */
   public SequenceLabeler(SequenceLabelerLearner learner) {
      this.encoderPair = learner.getEncoderPair();
      this.validator = learner.getValidator();
      this.preprocessors = learner.getPreprocessors().getModelProcessors();
      this.transitionFeatures = learner.getTransitionFeatures();
   }

   /**
    * Estimate double [ ].
    *
    * @param observation the observation
    * @param transitions the transitions
    * @return the double [ ]
    */
   public abstract double[] estimate(Iterator<Feature> observation, Iterator<String> transitions);


   @Override
   public EncoderPair getEncoderPair() {
      return encoderPair;
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
