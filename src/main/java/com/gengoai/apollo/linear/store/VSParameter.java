package com.gengoai.apollo.linear.store;

import com.gengoai.Parameters;
import com.gengoai.ValueTypeInformation;
import com.gengoai.apollo.linear.hash.LSHParameter;
import com.gengoai.conversion.Cast;

import java.lang.reflect.Type;

import static com.gengoai.reflection.Types.parameterizedType;


/**
 * @author David B. Bracewell
 */
public enum VSParameter implements ValueTypeInformation {
   CACHE_SIZE(Integer.class, 5_000),
   LOCATION(String.class, null),
   IN_MEMORY(Boolean.class, true),
   LSH(parameterizedType(Parameters.class, LSHParameter.class), null);

   private final Type type;
   private final Object defaultValue;

   VSParameter(Type type, Object defaultValue) {
      this.type = type;
      this.defaultValue = defaultValue;
   }

   @Override
   public Type getValueType() {
      return type;
   }

   @Override
   public <T> T defaultValue() {
      return Cast.as(defaultValue);
   }
}//END OF VSParams
