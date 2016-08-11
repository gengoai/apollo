package com.davidbracewell.apollo.linalg;

import lombok.NonNull;

import java.util.Collection;

/**
 * @author David B. Bracewell
 */
public enum VectorCompositions implements VectorComposition {
  Average {
    @Override
    public Vector compose(int dimension, @NonNull Collection<Vector> vectors) {
      if (vectors.size() == 0) {
        return new SparseVector(dimension);
      }
      Vector rval = new DenseVector(dimension);
      int count = 0;
      for (Vector vector : vectors) {
        if (vector.magnitude() > 0) {
          rval.addSelf(vector);
          count++;
        }
      }
      return rval.mapDivideSelf(count);
    }
  }
}//END OF VectorCompositions
