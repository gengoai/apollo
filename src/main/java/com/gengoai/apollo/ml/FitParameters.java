package com.gengoai.apollo.ml;

import com.gengoai.conversion.Cast;

/**
 * @author David B. Bracewell
 */
public class FitParameters extends Parameters {
   private static final long serialVersionUID = 1L;
   /**
    * The number of features in the problem
    */
   public int numFeatures;
   /**
    * The number of labels (regression problems will have 1)
    */
   public int numLabels;
   /**
    * Whether or not to be verbose when fitting and transforming with the model
    */
   public boolean verbose = false;
   /**
    * The interval (number of iterations) to output statistics about the learning process.
    */
   public int reportInterval = 100;

   @Override
   public FitParameters copy() {
      return Cast.as(super.copy());
   }


   @Override
   public FitParameters setParameter(String name, Object value) {
      return Cast.as(super.setParameter(name, value));
   }
}//END OF FitParameters
