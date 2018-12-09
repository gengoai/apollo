package com.gengoai.apollo.ml;

import com.gengoai.Copyable;
import com.gengoai.conversion.Val;
import com.gengoai.logging.Logger;
import com.gengoai.reflection.Reflect;
import com.gengoai.reflection.ReflectionException;

import java.io.Serializable;

/**
 * The type Parameters.
 *
 * @author David B. Bracewell
 */
public class Parameters implements Serializable, Copyable<Parameters> {
   private static final long serialVersionUID = 1L;
   private static final Logger logger = Logger.getLogger(Parameters.class);

   @Override
   public Parameters copy() {
      return Copyable.copy(this);
   }

   /**
    * Sets parameter.
    *
    * @param name  the name
    * @param value the value
    */
   public Parameters setParameter(String name, Object value) {
      try {
         Reflect.onObject(this).set(name, value);
      } catch (ReflectionException e) {
         if (e.getCause() instanceof NoSuchFieldException) {
            logger.warn("Invalid Parameter: {0}", name);
         } else {
            throw new RuntimeException(e);
         }
      }
      return this;
   }

   /**
    * Gets parameter.
    *
    * @param name the name
    * @return the parameter
    */
   public Val getParameter(String name) {
      try {
         return Val.of(Reflect.onObject(this).get(name).get());
      } catch (ReflectionException e) {
         logger.warn("Invalid Parameter: {0}", name);
         return Val.NULL;
      }
   }

}//END OF Parameters
