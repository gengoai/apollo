package com.davidbracewell.apollo.ml.classification;

import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.ml.EncoderPair;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.preprocess.PreprocessorList;
import com.davidbracewell.collection.counter.Counter;
import com.davidbracewell.collection.counter.Counters;
import lombok.NonNull;

import java.util.List;

/**
 * The type Ensemble.
 *
 * @author David B. Bracewell
 */
public class Ensemble extends Classifier {
   private static final long serialVersionUID = 1L;
   /**
    * The Models.
    */
   List<Classifier> models;

   /**
    * Instantiates a new Classifier.
    *
    * @param encoderPair   the encoder pair
    * @param preprocessors the preprocessors
    */
   protected Ensemble(@NonNull EncoderPair encoderPair, @NonNull PreprocessorList<Instance> preprocessors) {
      super(encoderPair, preprocessors);
   }


   @Override
   public Classification classify(@NonNull Instance instance) {
      Counter<String> results = Counters.newCounter();
      for (Classifier model : models) {
         results.increment(model.classify(instance).getResult());
      }
      results.divideBySum();
      return createResult(results);
   }

   @Override
   public Classification classify(Vector vector) {
      throw new IllegalAccessError();
   }

}// END OF Ensemble
