package com.davidbracewell.apollo.ml;

import lombok.NonNull;

import java.util.List;
import java.util.stream.Stream;

/**
 * <p> Encoders are responsible for encoding and decoding objects into double values. Encoders are used to create vector
 * representations of features. </p>
 *
 * @author David B. Bracewell
 */
public interface Encoder {

  /**
   * Encodes a stream of objects returning a double stream containing the encoded values.
   *
   * @param stream the stream to encode
   * @return the stream of encoded values
   */
  default void encode(@NonNull Stream<?> stream) {
    stream.forEach(this::encode);
  }

  /**
   * Encodes a single object returning its encoded value
   *
   * @param object the object
   * @return the encoded value
   */
  double encode(Object object);

  /**
   * Decodes an object given the encoded value
   *
   * @param value the encoded value
   * @return the object mapped to the encoded value
   */
  Object decode(double value);

  /**
   * Freezes the encoder restricting new objects from being mapped to values.
   */
  void freeze();

  /**
   * Unfreezes the encoder allowing new objects to be mapped to values.
   */
  void unFreeze();

  /**
   * Is the encoder currently frozen?
   *
   * @return True if frozen
   */
  boolean isFrozen();

  /**
   * The number of items that have mappings
   *
   * @return the number of items that have mappings
   */
  int size();

  /**
   * The values that have been encoded
   *
   * @return the list of values tha have been encoded
   */
  List<Object> values();

  /**
   * Creates new encoder of the same type as this one.
   *
   * @return the encoder
   */
  Encoder createNew();

}// END OF Encoder
