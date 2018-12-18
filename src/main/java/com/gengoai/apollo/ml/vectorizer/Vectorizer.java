package com.gengoai.apollo.ml.vectorizer;

import com.gengoai.Validation;
import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.apollo.ml.Example;
import com.gengoai.conversion.Cast;
import com.gengoai.json.JsonEntry;
import com.gengoai.json.JsonSerializable;
import com.gengoai.reflection.Reflect;

import java.io.Serializable;
import java.lang.reflect.Type;

/**
 * <p> Encoders are responsible for encoding and decoding objects into double values. Encoders are used to create
 * vector representations of features. </p>
 *
 * @param <T> the type parameter
 * @author David B. Bracewell
 */
public interface Vectorizer<T> extends JsonSerializable, Serializable {

   /**
    * Is label vectorizer boolean.
    *
    * @return the boolean
    */
   boolean isLabelVectorizer();

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

   /**
    * From json vectorizer.
    *
    * @param <T>        the type parameter
    * @param entry      the entry
    * @param parameters the parameters
    * @return the vectorizer
    */
   static <T> Vectorizer<T> fromJson(JsonEntry entry, Type... parameters) {
      Validation.checkState(entry.hasProperty("class"), "Serialized vectorizers must specify their class");
      try {
         return Cast.as(Reflect.onClass(entry.getStringProperty("class"))
                               .invoke("fromJson", entry, parameters)
                               .get());
      } catch (Exception e) {
         throw new RuntimeException(e);
      }
   }


}// END OF Encoder
