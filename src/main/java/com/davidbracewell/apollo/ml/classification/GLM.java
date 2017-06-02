package com.davidbracewell.apollo.ml.classification;

import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.ml.EncoderPair;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.preprocess.PreprocessorList;
import com.davidbracewell.apollo.optimization.Weights;
import com.davidbracewell.apollo.optimization.activation.Activation;

/**
 * @author David B. Bracewell
 */
public class GLM extends Classifier {
   private static final long serialVersionUID = 1L;
   Weights weights;
   Activation activation;

   /**
    * Instantiates a new Classifier.
    *
    * @param encoderPair   the pair of encoders to convert feature names into int/double values
    * @param preprocessors the preprocessors that the classifier will need apply at runtime
    */
   protected GLM(EncoderPair encoderPair, PreprocessorList<Instance> preprocessors) {
      super(encoderPair, preprocessors);
   }

   @Override
   public Classification classify(Vector vector) {
      double[] output;
      if (weights.isBinary()) {
         double score = activation.apply(weights.dot(vector)).get(0);
         output = new double[]{activation.isProbabilistic() ? 1d - score : -score, score};
      } else {
         output = activation.apply(weights.dot(vector)).toArray();
      }
      return new Classification(output, getLabelEncoder());
   }

}// END OF GLM
