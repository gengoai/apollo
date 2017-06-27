package com.davidbracewell.apollo.ml.nn;

import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.ml.classification.Classification;
import com.davidbracewell.apollo.ml.classification.Classifier;
import com.davidbracewell.apollo.ml.classification.ClassifierLearner;

/**
 * The type Feed forward network.
 *
 * @author David B. Bracewell
 */
public class SequentialNetwork extends Classifier {
   /**
    * The Layers.
    */
   Layer[] layers;

   /**
    * Instantiates a new Classifier.
    *
    * @param learner the learner
    */
   protected SequentialNetwork(ClassifierLearner learner) {
      super(learner);
   }

   @Override
   public Classification classify(Vector vector) {
      Vector m = vector;
      for (Layer layer : layers) {
         m = layer.forward(m);
      }
      return createResult(m.toArray());
   }

}// END OF SequentialNetwork
