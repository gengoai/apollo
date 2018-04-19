package com.gengoai.apollo.ml.sequence;

import com.gengoai.apollo.ml.Learner;
import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.apollo.ml.Learner;
import com.gengoai.apollo.ml.data.Dataset;
import lombok.Getter;
import lombok.NonNull;
import lombok.Setter;

/**
 * The type Sequence labeler learner.
 *
 * @author David B. Bracewell
 */
public abstract class SequenceLabelerLearner extends Learner<Sequence, SequenceLabeler> {
   private static final long serialVersionUID = 1L;
   @Getter
   @Setter
   protected Decoder decoder = new BeamDecoder(5);
   @Getter
   @Setter
   protected TransitionFeature transitionFeatures = TransitionFeature.FIRST_ORDER;
   @Getter
   @Setter
   protected SequenceValidator validator = SequenceValidator.ALWAYS_TRUE;

   @Override
   public SequenceLabeler train(@NonNull Dataset<Sequence> dataset) {
      dataset.encode();
      update(dataset.getEncoderPair(), dataset.getPreprocessors());
      transitionFeatures.fit(dataset);
      dataset.getEncoderPair().freeze();
      SequenceLabeler model = trainImpl(dataset);
      model.finishTraining();
      model.getFeatureEncoder().freeze();
      return model;
   }

}// END OF SequenceLabelerLearner
