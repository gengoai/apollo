package com.davidbracewell.apollo.linalg;

import lombok.NonNull;

import java.util.Arrays;
import java.util.Collection;

/**
 * <p>Defines a method of combining multiple vectors into one</p>
 *
 * @author David B. Bracewell
 */
@FunctionalInterface
public interface VectorComposition {

   /**
    * Compose the given vectors with given dimension into a single vector.
    *
    * @param dimension the dimension of the vectors
    * @param vectors   the vectors to compose
    * @return the composed vector
    */
   default Vector compose(int dimension, @NonNull Vector... vectors) {
      return compose(dimension, Arrays.asList(vectors));
   }

   /**
    * Compose the given vectors with given dimension into a single vector.
    *
    * @param dimension the dimension of the vectors
    * @param vectors   the vectors to compose
    * @return the composed vector
    */
   Vector compose(int dimension, Collection<Vector> vectors);

}// END OF VectorComposition
