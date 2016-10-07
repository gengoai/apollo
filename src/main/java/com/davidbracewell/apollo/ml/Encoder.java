package com.davidbracewell.apollo.ml;

import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.io.structured.StructuredSerializable;
import com.davidbracewell.stream.MStream;
import lombok.NonNull;

import java.util.List;
import java.util.stream.Stream;

/**
 * <p> Encoders are responsible for encoding and decoding objects into double values. Encoders are used to create vector
 * representations of features. </p>
 *
 * @author David B. Bracewell
 */
public interface Encoder extends StructuredSerializable {

   /**
    * Creates new encoder of the same type as this one.
    *
    * @return the encoder
    */
   Encoder createNew();

   /**
    * Decodes an object given the encoded value
    *
    * @param value the encoded value
    * @return the object mapped to the encoded value
    */
   Object decode(double value);

   /**
    * Encodes a stream of objects returning a double stream containing the encoded values.
    *
    * @param stream the stream to encode
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
    * Fit.
    *
    * @param dataset the dataset
    */
   void fit(Dataset<? extends Example> dataset);

   /**
    * Fit.
    *
    * @param stream the stream
    */
   void fit(MStream<String> stream);

   /**
    * Freezes the encoder restricting new objects from being mapped to values.
    */
   void freeze();

   /**
    * Get double.
    *
    * @param object the object
    * @return the double
    */
   double get(Object object);

   /**
    * Encodes a single object returning its encoded value as an index (int value)
    *
    * @param object the object to encode
    * @return the index (int value)
    */
   default int index(@NonNull Object object) {
      return (int) encode(object);
   }

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
    * Unfreezes the encoder allowing new objects to be mapped to values.
    */
   void unFreeze();

   /**
    * The values that have been encoded
    *
    * @return the list of values tha have been encoded
    */
   List<Object> values();

}// END OF Encoder
