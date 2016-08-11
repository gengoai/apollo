package com.davidbracewell.apollo.linalg;

import lombok.NonNull;

import java.util.Arrays;
import java.util.Collection;

/**
 * The interface Vector composition.
 *
 * @author David B. Bracewell
 */
@FunctionalInterface
public interface VectorComposition {

  /**
   * Compose vector.
   *
   * @param dimension the dimension
   * @param vectors   the vectors
   * @return the vector
   */
  default Vector compose(int dimension, @NonNull Vector... vectors) {
    return compose(dimension,Arrays.asList(vectors));
  }

  /**
   * Compose vector.
   *
   * @param dimension the dimension
   * @param vectors   the vectors
   * @return the vector
   */
  Vector compose(int dimension, Collection<Vector> vectors);

}// END OF VectorComposition
