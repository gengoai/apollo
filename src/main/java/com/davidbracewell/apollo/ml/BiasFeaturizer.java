package com.davidbracewell.apollo.ml;

/**
 * @author David B. Bracewell
 */
public class BiasFeaturizer<T> extends PredicateFeaturizer<T> {

   /**
    * Instantiates a new Predicate featurizer.
    */
   public BiasFeaturizer() {
      super("SPECIAL");
   }

   @Override
   public String extractPredicate(T t) {
      return "BIAS_FEATURE";
   }
}// END OF BiasFeaturizer
