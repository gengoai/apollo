package com.gengoai.apollo.ml;

import com.gengoai.Copyable;
import com.gengoai.conversion.Cast;
import com.gengoai.conversion.Val;
import com.gengoai.logging.Loggable;
import com.gengoai.reflection.Reflect;
import com.gengoai.reflection.ReflectionException;

import java.io.Serializable;
import java.lang.reflect.Field;

/**
 * <p>Generic interface for Named Arguments. Allows for set / get of public / protected fields using reflection. Usage
 * is to create a class that implements the interface defining the arguments and their default values as public fields
 * on the object. When able to directly use the implementing class, it is best to use the fields and avoid
 * reflection.</p>
 *
 * @param <T> the type parameter
 * @author David B. Bracewell
 */
public interface Parameters<T extends Parameters> extends Serializable, Copyable<T>, Loggable {


   @Override
   default T copy() {
      return Cast.as(Copyable.copy(this));
   }

   /**
    * Sets the parameter of the given name with the given value. Will print a warning if the parameter is not valid and
    * throw a RuntimeException for other reflection related errors.
    *
    * @param name  the parameter name
    * @param value the value to set.
    * @return the t
    */
   default T set(String name, Object value) {
      try {
         Reflect.onObject(this).set(name, value);
      } catch (ReflectionException e) {
         if (e.getCause() instanceof NoSuchFieldException) {
            logWarn("Invalid Parameter: {0}", name);
         } else {
            throw new RuntimeException(e);
         }
      }
      return Cast.as(this);
   }

   /**
    * Sets the parameter of the given name with the given value. Will print a warning if the parameter is not valid and
    * throw a RuntimeException for other reflection related errors.
    *
    * @param name the parameter name
    * @return the parameter value
    */
   default Val get(String name) {
      try {
         return Val.of(Reflect.onObject(this).get(name).get());
      } catch (ReflectionException e) {
         logWarn("Invalid Parameter: {0}", name);
         return Val.NULL;
      }
   }


   /**
    * Helper method to generate a String representation of this object using reflection on the fields and their values.
    *
    * @return the string representation of this object.
    */
   default String asString() {
      StringBuilder builder = new StringBuilder(this.getClass().getSimpleName()).append("{");
      int cnt = 0;
      for (Field f : Reflect.onObject(this).getFields()) {
         try {
            if (cnt > 0) {
               builder.append(", ");
            }
            builder.append(f.getName())
                   .append("=")
                   .append(f.get(this));
            cnt++;
         } catch (IllegalAccessException e) {
            //ignore
         }
      }
      return builder.append("}").toString();
   }


}//END OF Parameters
