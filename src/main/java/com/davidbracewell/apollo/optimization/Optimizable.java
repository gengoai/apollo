package com.davidbracewell.apollo.optimization;

import com.davidbracewell.apollo.linalg.Vector;

/**
 * The interface Optimizable.
 *
 * @author David B. Bracewell
 */
public interface Optimizable {

   /**
    * Gets parameters.
    *
    * @return the parameters
    */
   Vector getParameters();

   /**
    * Sets parameters.
    *
    * @param parameters the parameters
    */
   void setParameters(Vector parameters);

   /**
    * Gets value.
    *
    * @return the value
    */
   double getValue();

   void iterate(Vector v);


   /**
    * The interface By gradient.
    */
   interface ByGradient extends Optimizable {

      /**
       * Gets gradient.
       *
       * @return the gradient
       */
      Vector getGradient();

   }


}//END OF Optimizable
