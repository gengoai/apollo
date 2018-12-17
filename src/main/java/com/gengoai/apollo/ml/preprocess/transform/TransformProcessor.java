package com.gengoai.apollo.ml.preprocess.transform;

import com.gengoai.apollo.ml.preprocess.Preprocessor;

/**
 * @author David B. Bracewell
 */
public interface TransformProcessor extends Preprocessor {

   @Override
   default boolean trainOnly() {
      return false;
   }


}//END OF TransformProcessor
