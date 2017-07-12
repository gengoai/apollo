package com.davidbracewell.apollo.ml.classification;

import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.optimization.WeightMatrix;
import com.davidbracewell.apollo.optimization.activation.Activation;
import com.davidbracewell.collection.counter.MultiCounter;
import com.davidbracewell.collection.counter.MultiCounters;

/**
 * @author David B. Bracewell
 */
public class GeneralizedLinearModel extends Classifier {
   private static final long serialVersionUID = 1L;
   WeightMatrix weights;
   Activation activation;

   protected GeneralizedLinearModel(ClassifierLearner learner) {
      super(learner);
   }


   @Override
   public Classification classify(Vector vector) {
      return createResult(weights.dot(vector, activation).toArray());
   }

   @Override
   public MultiCounter<String, String> getModelParameters() {
      MultiCounter<String, String> weightCntr = MultiCounters.newMultiCounter();
//      for (int fi = 0; fi < getFeatureEncoder().size(); fi++) {
//         String featureName = decodeFeature(fi).toString();
//         for (int li = 0; li < getLabelEncoder().size(); li++) {
//            String label = decodeLabel(li).toString();
//            weightCntr.set(featureName, label, weights.getTheta().get(li, fi));
//         }
//      }
//      for (int i = 0; i < getLabelEncoder().size(); i++) {
//         String label = decodeLabel(i).toString();
//         weightCntr.set("***BIAS***", label, weights.getBias().get(i));
//      }
      return weightCntr;
   }
}// END OF GLM
