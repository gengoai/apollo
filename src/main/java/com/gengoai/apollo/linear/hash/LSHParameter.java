package com.gengoai.apollo.linear.hash;

import com.gengoai.NamedParameters;
import com.gengoai.conversion.Cast;

import java.lang.reflect.Type;

/**
 * @author David B. Bracewell
 */
public enum LSHParameter implements NamedParameters.Value {
   BANDS(Integer.class, 5),
   BUCKETS(Integer.class, 20),
   SIGNATURE(String.class, "COSINE"),
   DIMENSION(Integer.class, 1),
   SIGNATURE_SIZE(Integer.class, 1),
   MAX_W(Integer.class, 100),
   THRESHOLD(Double.class, 0.5);

   private final Type type;
   private final Object defaultValue;

   LSHParameter(Type type, Object defaultValue) {
      this.type = type;
      this.defaultValue = defaultValue;
   }

   @Override
   public Type getValueType() {
      return type;
   }

   @Override
   public Object defaultValue() {
      return Cast.as(defaultValue);
   }

}//END OF LSHParameters
