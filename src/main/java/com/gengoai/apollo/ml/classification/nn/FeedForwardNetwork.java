package com.gengoai.apollo.ml.classification.nn;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.linear.NDArrayFactory;
import com.gengoai.apollo.ml.Instance;
import com.gengoai.apollo.ml.classification.Classification;
import com.gengoai.apollo.ml.classification.Classifier;
import com.gengoai.apollo.ml.classification.ClassifierLearner;
import com.gengoai.apollo.ml.encoder.EncoderPair;
import com.gengoai.apollo.ml.optimization.activation.Activation;
import com.gengoai.apollo.ml.preprocess.PreprocessorList;
import com.gengoai.conversion.Cast;
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

   public Layer getLayer(int i) {
      return layers.get(i);
   }

   @Override
   public Classification classify(NDArray vector) {
      for (Layer layer : layers) {
         vector = layer.forward(vector);
      }
      if (vector.length() == 1) {
         Activation activation = Cast.<WeightLayer>as(layers.get(layers.size() - 1)).activation;
         double shift = activation.isProbabilistic() ? 1d : 0d;
         vector = NDArrayFactory.wrap(new double[]{shift - vector.scalarValue(), vector.scalarValue()});
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
