package com.gengoai.apollo.ml.classification;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.ml.Instance;
import com.gengoai.collection.counter.Counter;
import com.gengoai.collection.counter.Counters;
import com.gengoai.apollo.ml.Instance;
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
