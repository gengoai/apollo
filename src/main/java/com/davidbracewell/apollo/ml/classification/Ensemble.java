package com.davidbracewell.apollo.ml.classification;

import com.davidbracewell.apollo.linear.NDArray;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.collection.counter.Counter;
import com.davidbracewell.collection.counter.Counters;
import lombok.NonNull;

import java.util.List;

/**
 * A classifier made up of multiple "weak" classifiers that are combined using a majority vote strategy.
 *
 * @author David B. Bracewell
 */
public class Ensemble extends Classifier {
   private static final long serialVersionUID = 1L;
   List<Classifier> models;

   protected Ensemble(ClassifierLearner learner) {
      super(learner);
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
   public Classification classify(NDArray vector) {
      throw new IllegalAccessError();
   }

}// END OF Ensemble
