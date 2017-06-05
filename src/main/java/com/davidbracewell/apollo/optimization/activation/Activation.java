package com.davidbracewell.apollo.optimization.activation;

import com.davidbracewell.apollo.linalg.Vector;
import lombok.NonNull;

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
    * Apply vector.
    *
    * @param x the x
    * @return the vector
    */
   default Vector apply(@NonNull Vector x) {
      return x.map(this::apply);
   }

   /**
    * Is probabilistic boolean.
    *
    * @return the boolean
    */
   default boolean isProbabilistic() {
      return false;
   }


}//END OF Activation
