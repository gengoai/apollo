package com.davidbracewell.apollo.ml.preprocess.transform;

import com.davidbracewell.apollo.ml.Example;
import com.davidbracewell.apollo.ml.preprocess.Preprocessor;

/**
 * Specialized processor that transforms feature values
 *
 * @param <T> the example type parameter
 * @author David B. Bracewell
 */
public interface TransformProcessor<T extends Example> extends Preprocessor<T> {

   @Override
   default boolean trainOnly() {
      return false;
   }

}// END OF TransformProcessor
