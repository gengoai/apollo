package com.davidbracewell.apollo.ml.sequence;

import com.davidbracewell.apollo.ml.Feature;
import com.davidbracewell.apollo.ml.FeatureVector;
import com.davidbracewell.apollo.ml.classification.Classifier;
import lombok.NonNull;

import java.util.Iterator;

/**
 * <p>Greedy sequence labeler that wraps a standard {@link Classifier} and optimizing for local instead of global
 * structure.</p>
 *
 * @author David B. Bracewell
 */
public class WindowedLabeler extends SequenceLabeler {
   private static final long serialVersionUID = 1L;
   /**
    * The Classifier.
    */
   Classifier classifier;


   /**
    * Instantiates a new Model.
    *
    * @param learner the learner
    */
   public WindowedLabeler(@NonNull WindowedLearner learner) {
      super(learner);
      super.setDecoder(new WindowDecoder());
   }

   @Override
   public double[] estimate(Iterator<Feature> observation, Iterator<String> transitions) {
      FeatureVector vector = new FeatureVector(classifier.getEncoderPair());
      observation.forEachRemaining(vector::set);
      transitions.forEachRemaining(t -> vector.set(t, 1.0));
      return classifier.classify(vector).distribution();
   }

   @Override
   public void setDecoder(@NonNull Decoder decoder) {
      throw new UnsupportedOperationException();
   }

}// END OF WindowedLabeler
