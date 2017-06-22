package com.davidbracewell.apollo.ml.nn;

import com.davidbracewell.apollo.linalg.Matrix;
import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.ml.classification.Classification;
import com.davidbracewell.apollo.ml.classification.Classifier;
import com.davidbracewell.apollo.ml.classification.ClassifierLearner;

/**
 * The type Feed forward network.
 *
 * @author David B. Bracewell
 */
public class FeedForwardNetwork extends Classifier {
   /**
    * The Layers.
    */
   Layer[] layers;

   /**
    * Instantiates a new Classifier.
    *
    * @param learner the learner
    */
   protected FeedForwardNetwork(ClassifierLearner learner) {
      super(learner);
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
