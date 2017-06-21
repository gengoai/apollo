package com.davidbracewell.apollo.linalg;

import com.davidbracewell.guava.common.base.Preconditions;

import java.io.Serializable;

/**
 * @author David B. Bracewell
 */
public abstract class BaseVector implements Vector, Serializable {
   private static final long serialVersionUID = 1L;

   @Override
   public Vector compress() {
      return this;
   }

   @Override
   public Vector copy() {
      Vector vPrime = createNew(dimension());
      forEach(e -> vPrime.set(e.index, e.value));
      return vPrime;
   }

   protected abstract Vector createNew(int dimension);

   @Override
   public boolean isDense() {
      return false;
   }

   @Override
   public boolean isSparse() {
      return false;
   }

   @Override
   public Vector redim(int newDimension) {
      Preconditions.checkArgument(newDimension > 0, "Only positive dimensions allowed.");
      Vector vPrime = createNew(newDimension);
      for (int i = 0; i < newDimension; i++) {
         vPrime.set(i, get(i));
      }
      return vPrime;
   }

   @Override
   public Vector slice(int from, int to) {
      Preconditions.checkPositionIndex(from, dimension());
      Preconditions.checkPositionIndex(to, dimension() + 1);
      Preconditions.checkState(to > from, "To index must be > from index");
      Vector vPrime = createNew(to - from);
      for (int i = from; i < to; i++) {
         vPrime.set(i - from, get(i));
      }
      return vPrime;
   }

   @Override
   public double[] toArray() {
      double[] array = new double[dimension()];
      forEach(e -> array[e.getIndex()] = e.getValue());
      return array;
   }

   @Override
   public Vector zero() {
      return createNew(dimension());
   }


}// END OF BaseVector
