package com.gengoai.apollo.ml.vectorizer;

import com.gengoai.Validation;
import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.ml.Dataset;
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
 * @author David B. Bracewell
 */
public interface Vectorizer<T> extends JsonSerializable, Serializable {

   boolean isLabelVectorizer();

   T decode(double value);

   double encode(T value);

   void fit(Dataset dataset);

   int size();

   NDArray transform(Example example);

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
