package com.davidbracewell.apollo.linalg;

import lombok.NonNull;

import java.util.Arrays;
import java.util.Collection;

/**
 * @author David B. Bracewell
 */
@FunctionalInterface
public interface VectorComposition {

  default Vector compose(int dimension, @NonNull Vector... vectors) {
    return compose(dimension,Arrays.asList(vectors));
  }

  Vector compose(int dimension, Collection<Vector> vectors);

}// END OF VectorComposition
