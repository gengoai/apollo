package com.gengoai.apollo.ml;

import com.gengoai.Parameters;

/**
 * @author David B. Bracewell
 */
public class FitParameters implements Parameters<FitParameters> {
   private static final long serialVersionUID = 1L;
   /**
    * Whether or not to be verbose when fitting and transforming with the model
    */
   public boolean verbose = false;


   @Override
   public String toString() {
      return asString();
   }
}//END OF FitParameters
