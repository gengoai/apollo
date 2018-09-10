package com.gengoai.apollo.linear.store;

import com.gengoai.Parameters;
import com.gengoai.ValueTypeInformation;
import com.gengoai.apollo.hash.LSHParameter;
import com.gengoai.conversion.Cast;
import com.google.gson.reflect.TypeToken;

import java.lang.reflect.Type;

/**
 * @author David B. Bracewell
 */
public enum VSParams implements ValueTypeInformation {
   CACHE_SIZE(Integer.class, 5_000),
   LOCATION(String.class, null),
   LSH(new TypeToken<Parameters<LSHParameter>>(){{}}.getType(), null);

   private final Type type;
   private final Object defaultValue;

   VSParams(Type type, Object defaultValue) {
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
