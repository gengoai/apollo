package com.davidbracewell.apollo.ml.preprocess;

import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.apollo.ml.Example;

import java.io.Serializable;

/**
 * The interface Preprocessor.
 *
 * @param <T> the type parameter
 * @author David B. Bracewell
 */
public interface Preprocessor<T extends Example> extends Serializable {

  /**
   * Process dataset.
   *
   * @param dataset the dataset
   * @return the dataset
   */
  void fit(Dataset<T> dataset);

  /**
   * Train only boolean.
   *
   * @return the boolean
   */
  boolean trainOnly();

  /**
   * Reset.
   */
  void reset();


  /**
   * Apply t.
   *
   * @param example the example
   * @return the t
   */
  T apply(T example);


  /**
   * Describe string.
   *
   * @return the string
   */
  String describe();

}//END OF Preprocessor
