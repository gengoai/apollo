package com.davidbracewell.apollo.optimization.activation;

import com.davidbracewell.apollo.linalg.Vector;
import lombok.NonNull;

/**
 * @author David B. Bracewell
 */
public interface DifferentiableActivation extends Activation {

   default Vector gradient(@NonNull Vector in) {
      return valueGradient(apply(in));
   }

   default double gradient(double in) {
      return valueGradient(apply(in));
   }

   double valueGradient(double activated);

   default Vector valueGradient(@NonNull Vector activated) {
      return activated.map(this::valueGradient);
   }

}// END OF DifferentiableActivation
