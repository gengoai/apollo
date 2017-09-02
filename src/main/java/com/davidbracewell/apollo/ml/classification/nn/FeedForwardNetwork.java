package com.davidbracewell.apollo.ml.classification.nn;

import com.davidbracewell.apollo.linalg.DenseFloatMatrix;
import com.davidbracewell.apollo.linalg.Matrix;
import com.davidbracewell.apollo.linalg.Vector;
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
   public Classification classify(Vector vector) {
      Matrix m = new DenseFloatMatrix(vector.dimension(), 1, vector.toFloatArray());
      for (Layer layer : layers) {
         m = layer.forward(m);
      }
      return createResult(m.toDoubleArray());
   }

}// END OF FeedForwardNetwork
