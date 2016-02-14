package com.davidbracewell.apollo.ml.preprocess;

import com.davidbracewell.apollo.ml.Encoder;
import com.davidbracewell.apollo.ml.Example;

/**
 * The interface Preprocessor.
 *
 * @param <T> the type parameter
 * @author David B. Bracewell
 */
public interface Preprocessor<T extends Example> {

  /**
   * Visit.
   *
   * @param example the example
   */
  void visit(T example);

  /**
   * Process t.
   *
   * @param example the example
   * @return the t
   */
  T process(T example);

  /**
   * Finish.
   */
  void finish();

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
   * Trim to size.
   *
   * @param encoder the encoder
   */
  void trimToSize(Encoder encoder);

  String describe();


}//END OF Preprocessor
