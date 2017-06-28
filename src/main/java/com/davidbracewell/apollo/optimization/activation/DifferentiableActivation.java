package com.davidbracewell.apollo.optimization.activation;

import com.davidbracewell.apollo.linalg.Matrix;
import com.davidbracewell.apollo.linalg.Vector;
import lombok.NonNull;

/**
 * @author David B. Bracewell
 */
public interface DifferentiableActivation extends Activation {

   Vector valueGradient(Vector activated);

   /**
    * Apply derivative vector.
    *
    * @param predicted the predicted
    * @param actual    the actual
    * @return the vector
    */
   Vector valueGradient(@NonNull Vector predicted, @NonNull Vector actual);

   /**
    * Apply derivative vector.
    *
    * @param predicted the predicted
    * @param actual    the actual
    * @return the vector
    */
   Matrix valueGradient(@NonNull Matrix predicted, @NonNull Matrix actual);
}// END OF DifferentiableActivation
