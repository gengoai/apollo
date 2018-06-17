package com.gengoai.apollo.ml.preprocess.filter;

import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.preprocess.Preprocessor;

/**
 * Specialized processor that removes features based on a given criteria.
 *
 * @param <T> the example type parameter
 * @author David B. Bracewell
 */
public interface FilterProcessor<T extends Example> extends Preprocessor<T> {

   @Override
   default boolean trainOnly() {
      return true;
   }

}//END OF FilterProcessor
