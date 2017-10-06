package com.davidbracewell.apollo.ml.classification.nn;

import com.davidbracewell.apollo.linear.NDArray;
import com.davidbracewell.apollo.ml.classification.Classification;
import com.davidbracewell.apollo.ml.classification.Classifier;
import com.davidbracewell.apollo.ml.classification.ClassifierLearner;

import java.util.ArrayList;

/**
 * @author David B. Bracewell
 */
public class FeedForwardNetwork extends Classifier {
   /**
    * The Layers.
    */
   ArrayList<Layer> layers;

   /**
    * Instantiates a new Classifier.
    *
    * @param learner the learner
    */
   protected FeedForwardNetwork(ClassifierLearner learner) {
      super(learner);
   }

   @Override
   public Classification classify(NDArray vector) {
      for (Layer layer : layers) {
         vector = layer.forward(vector);
      }
      return createResult(vector.toArray());
   }

}// END OF FeedForwardNetwork
