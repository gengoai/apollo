package com.gengoai.apollo.ml.preprocess;

import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.Feature;
import com.gengoai.function.SerializableFunction;

/**
 * @author David B. Bracewell
 */
public interface FeaturePreprocessor extends Preprocessor, SerializableFunction<Feature, Feature> {

   @Override
   default Example apply(Example example) {
      return example.mapFeatures(this);
   }

   @Override
   default Feature apply(Feature in) {
      if (requirePreprocessing(in)) {
         return preprocess(in);
      }
      return in.copy();
   }


   Feature preprocess(Feature in);

   boolean requirePreprocessing(Feature f);


}//END OF FeaturePreprocessor
