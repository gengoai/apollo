package com.gengoai.apollo.ml.neural;

import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.FitParameters;
import com.gengoai.apollo.ml.classification.Classification;
import com.gengoai.apollo.ml.classification.Classifier;
import com.gengoai.apollo.ml.data.Dataset;

import java.util.ArrayList;

/**
 * @author David B. Bracewell
 */
public class FeedForwardNetwork extends Classifier {
   private static final long serialVersionUID = 1L;
   ArrayList<Layer> layers = new ArrayList<>();

   @Override
   protected Classifier fitPreprocessed(Dataset preprocessed, FitParameters fitParameters) {
      return this;
   }

   @Override
   public FitParameters getDefaultFitParameters() {
      return null;
   }

   @Override
   public Classification predict(Example example) {
      return null;
   }

}//END OF FeedForwardNetwork
