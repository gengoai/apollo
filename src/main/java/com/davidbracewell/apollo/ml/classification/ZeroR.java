package com.davidbracewell.apollo.ml.classification;

import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.ml.EncoderPair;
import com.davidbracewell.apollo.ml.preprocess.PreprocessorList;
import com.davidbracewell.collection.counter.HashMapMultiCounter;
import com.davidbracewell.collection.counter.MultiCounter;
import lombok.NonNull;

/**
 * Baseline classifier that always predicts the majority class
 *
 * @author David B. Bracewell
 */
public class ZeroR extends Classifier {
   private static final long serialVersionUID = 1L;
   double[] distribution;

   /**
    * Instantiates a new ZeroR classifier.
    *
    * @param encoderPair the encoder pair
    */
   protected ZeroR(@NonNull EncoderPair encoderPair) {
      super(encoderPair, PreprocessorList.empty());
   }

   @Override
   public Classification classify(Vector vector) {
      return new Classification(distribution, getLabelEncoder());
   }

   @Override
   public MultiCounter<String, String> getModelParameters() {
      MultiCounter<String, String> weights = new HashMapMultiCounter<>();
      for (int i = 0; i < distribution.length; i++) {
         weights.set("*", getLabelEncoder().decode(i).toString(), distribution[i]);
      }
      return weights;
   }

}// END OF ZeroR
