package com.davidbracewell.apollo.ml.preprocess;

import com.davidbracewell.apollo.ml.Example;

/**
 * @author David B. Bracewell
 */
public interface TransformProcessor<T extends Example> extends Preprocessor<T> {

  @Override
  default boolean trainOnly() {
    return false;
  }

}// END OF TransformProcessor
