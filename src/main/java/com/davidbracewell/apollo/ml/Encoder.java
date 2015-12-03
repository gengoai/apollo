package com.davidbracewell.apollo.ml;

import lombok.NonNull;

import java.util.List;
import java.util.stream.DoubleStream;
import java.util.stream.Stream;

/**
 * The interface Encoder.
 *
 * @author David B. Bracewell
 */
public interface Encoder {

  /**
   * Encode double stream.
   *
   * @param stream the stream
   * @return the double stream
   */
  default DoubleStream encode(@NonNull Stream<?> stream) {
    return stream.mapToDouble(this::encode);
  }

  /**
   * Encode double.
   *
   * @param object the object
   * @return the double
   */
  double encode(Object object);

  /**
   * Decode object.
   *
   * @param value the value
   * @return the object
   */
  Object decode(double value);

  /**
   * Freeze.
   */
  void freeze();

  /**
   * Un freeze.
   */
  void unFreeze();

  /**
   * Is frozen boolean.
   *
   * @return the boolean
   */
  boolean isFrozen();

  /**
   * Size int.
   *
   * @return the int
   */
  int size();

  /**
   * Values list.
   *
   * @return the list
   */
  List<Object> values();

  Encoder createNew();

}// END OF Encoder
