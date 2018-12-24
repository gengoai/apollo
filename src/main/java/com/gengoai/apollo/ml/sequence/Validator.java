package com.gengoai.apollo.ml.sequence;

import com.gengoai.apollo.ml.Example;

import java.io.Serializable;

/**
 * The interface Validator.
 *
 * @author David B. Bracewell
 */
public interface Validator extends Serializable {

   /**
    * The constant ALWAYS_TRUE.
    */
   Validator ALWAYS_TRUE = (Validator) (currentLabel, previousLabel, instance) -> true;

   /**
    * Is valid boolean.
    *
    * @param currentLabel  the current label
    * @param previousLabel the previous label
    * @param instance      the instance
    * @return the boolean
    */
   boolean isValid(String currentLabel, String previousLabel, Example instance);

}//END OF Validator
