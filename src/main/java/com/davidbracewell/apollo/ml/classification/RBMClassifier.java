package com.davidbracewell.apollo.ml.classification;

import com.davidbracewell.apollo.linalg.SparseVector;
import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.ml.nn.BernoulliRBM;

/**
 * @author David B. Bracewell
 */
public class RBMClassifier extends Classifier {
   BernoulliRBM rbm;
   Classifier classifier;

   protected RBMClassifier(ClassifierLearner learner) {
      super(learner);
   }


   @Override
   public Classification classify(Vector vector) {
      Vector sparse = SparseVector.zeros(rbm.getNumHidden());
      rbm.runVisibleProbs(vector)
         .nonZeroIterator()
         .forEachRemaining(e -> sparse.set(e.index, e.getValue()));
      return classifier.classify(sparse);
   }
}// END OF RBMClassifier
