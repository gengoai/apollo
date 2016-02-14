package com.davidbracewell.apollo.ml.preprocess.transform;

import com.davidbracewell.apollo.ml.Example;
import com.davidbracewell.apollo.ml.preprocess.Preprocessor;

/**
 * @author David B. Bracewell
 */
public interface TransformProcessor<T extends Example> extends Preprocessor<T> {

  @Override
  default boolean trainOnly() {
    return false;
  }

}// END OF TransformProcessor
