package com.davidbracewell.apollo.ml.preprocess;

/**
 * The interface Restricted.
 *
 * @author David B. Bracewell
 */
public interface Restricted {

   /**
    * Accept all boolean.
    *
    * @return the boolean
    */
   boolean acceptAll();

   /**
    * Filter positive boolean.
    *
    * @param featureName the feature name
    * @return the boolean
    */
   default boolean filterPositive(String featureName) {
      return acceptAll() || featureName.startsWith(getRestriction());
   }

   /**
    * Gets restriction.
    *
    * @return the restriction
    */
   String getRestriction();

}//END OF Restricted
