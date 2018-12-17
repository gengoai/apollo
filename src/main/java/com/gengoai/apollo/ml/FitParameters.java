package com.gengoai.apollo.ml;

/**
 * @author David B. Bracewell
 */
public class FitParameters implements Parameters<FitParameters> {
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
   public String toString() {
      return asString();
   }
}//END OF FitParameters
