package com.davidbracewell.apollo.optimization;

import java.io.Serializable;

/**
 * The interface Activation.
 *
 * @author David B. Bracewell
 */
public interface Activation extends Serializable {

   /**
    * Apply double.
    *
    * @param x the x
    * @return the double
    */
   double apply(double x);

   /**
    * Gradient double.
    *
    * @param x the x
    * @return the double
    */
   double gradient(double x);

   /**
    * Value gradient double.
    *
    * @param x the x
    * @return the double
    */
   double valueGradient(double x);


}//END OF Activation
