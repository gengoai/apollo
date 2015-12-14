package com.davidbracewell.apollo.ml.preprocess.filter;

import com.davidbracewell.apollo.ml.Encoder;
import com.davidbracewell.apollo.ml.Example;
import com.davidbracewell.apollo.ml.preprocess.Preprocessor;

/**
 * @author David B. Bracewell
 */
public interface FilterProcessor<T extends Example> extends Preprocessor<T> {

  @Override
  default boolean trainOnly() {
    return true;
  }

  @Override
  default void trimToSize(Encoder encoder) {
  }

}//END OF FilterProcessor
