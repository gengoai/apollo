package com.davidbracewell.apollo.ml.classification;

import com.davidbracewell.apollo.linalg.SparseVector;
import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.ml.nn.BernoulliRBM;
import com.davidbracewell.apollo.ml.EncoderPair;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.preprocess.PreprocessorList;

/**
 * @author David B. Bracewell
 */
public class RBMClassifier extends Classifier {
   BernoulliRBM rbm;
   Classifier classifier;

   /**
    * Instantiates a new Classifier.
    *
    * @param encoderPair   the pair of encoders to convert feature names into int/double values
    * @param preprocessors the preprocessors that the classifier will need apply at runtime
    */
   protected RBMClassifier(EncoderPair encoderPair, PreprocessorList<Instance> preprocessors) {
      super(encoderPair, preprocessors);
   }

   @Override
   public Classification classify(Vector vector) {
      Vector sparse = SparseVector.zeros(rbm.getNumHidden());
      rbm.runVisible(vector)
         .nonZeroIterator()
         .forEachRemaining(e -> sparse.set(e.index, e.getValue()));
      return classifier.classify(sparse);
   }
}// END OF RBMClassifier
