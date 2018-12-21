package com.gengoai.apollo.ml.vectorizer;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.data.Dataset;

import java.io.Serializable;
import java.util.Collections;
import java.util.Set;

/**
 * <p> Encoders are responsible for encoding and decoding objects into double values. Encoders are used to create
 * vector representations of features. </p>
 *
 * @param <T> the type parameter
 * @author David B. Bracewell
 */
public interface Vectorizer<T> extends Serializable {

   default Set<String> alphabet() {
      return Collections.emptySet();
   }

   /**
    * Decode t.
    *
    * @param value the value
    * @return the t
    */
   T decode(double value);

   /**
    * Encode double.
    *
    * @param value the value
    * @return the double
    */
   double encode(T value);

   /**
    * Fit.
    *
    * @param dataset the dataset
    */
   void fit(Dataset dataset);

   /**
    * Size int.
    *
    * @return the int
    */
   int size();

   /**
    * Transform nd array.
    *
    * @param example the example
    * @return the nd array
    */
   NDArray transform(Example example);


}// END OF Encoder
