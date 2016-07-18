package com.davidbracewell.apollo.ml.preprocess;

/**
 * @author David B. Bracewell
 */
public interface Restricted {

  boolean acceptAll();

  String getRestriction();

  default boolean filterPositive(String featureName) {
    return acceptAll() || featureName.startsWith(getRestriction());
  }


}//END OF Restricted
