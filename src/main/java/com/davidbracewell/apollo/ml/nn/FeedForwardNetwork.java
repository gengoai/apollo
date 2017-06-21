package com.davidbracewell.apollo.ml.nn;

import com.davidbracewell.apollo.linalg.Matrix;
import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.ml.EncoderPair;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.classification.Classification;
import com.davidbracewell.apollo.ml.classification.Classifier;
import com.davidbracewell.apollo.ml.preprocess.PreprocessorList;

/**
 * @author David B. Bracewell
 */
public class FeedForwardNetwork extends Classifier {
   Layer[] layers;

   /**
    * Instantiates a new Classifier.
    *
    * @param encoderPair   the pair of encoders to convert feature names into int/double values
    * @param preprocessors the preprocessors that the classifier will need apply at runtime
    */
   protected FeedForwardNetwork(EncoderPair encoderPair, PreprocessorList<Instance> preprocessors) {
      super(encoderPair, preprocessors);
   }

   @Override
   public Classification classify(Vector vector) {
      Matrix m = vector.toMatrix();
      for (Layer layer : layers) {
         m = layer.forward(m);
      }
      return createResult(m.row(0).toArray());
   }

}// END OF FeedForwardNetwork
