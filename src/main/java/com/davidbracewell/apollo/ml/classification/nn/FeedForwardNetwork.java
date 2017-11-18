package com.davidbracewell.apollo.ml.classification.nn;

import com.davidbracewell.apollo.linear.NDArray;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.classification.Classification;
import com.davidbracewell.apollo.ml.classification.Classifier;
import com.davidbracewell.apollo.ml.classification.ClassifierLearner;
import com.davidbracewell.apollo.ml.encoder.EncoderPair;
import com.davidbracewell.apollo.ml.preprocess.PreprocessorList;
import lombok.NonNull;

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

   protected FeedForwardNetwork(@NonNull PreprocessorList<Instance> preprocessors, EncoderPair encoderPair) {
      super(preprocessors, encoderPair);
   }

   public Layer getLayer(int i){
      return layers.get(i);
   }

   @Override
   public Classification classify(NDArray vector) {
      for (Layer layer : layers) {
         vector = layer.forward(vector);
      }
      return createResult(vector.toArray());
   }

   public FeedForwardNetwork copy() {
      FeedForwardNetwork ffn = new FeedForwardNetwork(getPreprocessors(), getEncoderPair());
      ffn.layers = new ArrayList<>();
      layers.forEach(l -> ffn.layers.add(l.copy()));
      return ffn;
   }

}// END OF FeedForwardNetwork
