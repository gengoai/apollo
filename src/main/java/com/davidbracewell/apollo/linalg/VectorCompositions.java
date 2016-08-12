package com.davidbracewell.apollo.linalg;

import lombok.NonNull;

import java.util.Arrays;
import java.util.Collection;

/**
 * The enum Vector compositions.
 *
 * @author David B. Bracewell
 */
public enum VectorCompositions implements VectorComposition {
  /**
   * The Average.
   */
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
  },
  /**
   * The Sum.
   */
  Sum {
    @Override
    public Vector compose(int dimension, @NonNull Collection<Vector> vectors) {
      if (vectors.size() == 0) {
        return new SparseVector(dimension);
      }
      Vector rval = new DenseVector(dimension);
      vectors.forEach(rval::addSelf);
      return rval;
    }
  },
  /**
   * The Point wise multiply.
   */
  PointWiseMultiply {
    @Override
    public Vector compose(int dimension, @NonNull Collection<Vector> vectors) {
      if (vectors.size() == 0) {
        return new SparseVector(dimension);
      }
      Vector rval = DenseVector.ones(dimension);
      vectors.stream().filter(vector -> vector.magnitude() > 0).forEach(rval::multiplySelf);
      return rval;
    }
  },
  /**
   * The Max.
   */
  Max {
    @Override
    public Vector compose(int dimension, @NonNull Collection<Vector> vectors) {
      if (vectors.size() == 0) {
        return new SparseVector(dimension);
      }
      double[] values = new double[dimension];
      Arrays.fill(values, Double.NEGATIVE_INFINITY);
      Vector rval = new DenseVector(values);
      vectors.stream().filter(vector -> vector.magnitude() > 0)
             .forEach(v -> v.forEach(e -> {
                        if (e.getValue() > rval.get(e.getIndex())) {
                          rval.set(e.getIndex(), e.getValue());
                        }
                      })
             );
      return rval;
    }
  },
  /**
   * The Min.
   */
  Min {
    @Override
    public Vector compose(int dimension, @NonNull Collection<Vector> vectors) {
      if (vectors.size() == 0) {
        return new SparseVector(dimension);
      }
      double[] values = new double[dimension];
      Arrays.fill(values, Double.POSITIVE_INFINITY);
      Vector rval = new DenseVector(values);
      vectors.stream().filter(vector -> vector.magnitude() > 0)
             .forEach(v -> v.forEach(e -> {
                        if (e.getValue() < rval.get(e.getIndex())) {
                          rval.set(e.getIndex(), e.getValue());
                        }
                      })
             );
      return rval;
    }
  }

}//END OF VectorCompositions
