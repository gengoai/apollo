package com.davidbracewell.apollo.linalg;

import com.davidbracewell.conversion.Cast;

import java.util.Arrays;

/**
 * @author David B. Bracewell
 */
class ColumnVector extends BaseVector {
   private static final long serialVersionUID = 1L;
   final int column;
   private Matrix entries;

   ColumnVector(Matrix entries, int column) {
      this.entries = entries;
      this.column = column;
   }

   @Override
   public Vector compress() {
      return this;
   }

   @Override
   public Vector copy() {
      return new DenseVector(toArray());
   }

   @Override
   protected Vector createNew(int dimension) {
      return entries.isSparse() ? Vector.sZeros(dimension) : Vector.dZeros(dimension);
   }

   @Override
   public int dimension() {
      return entries.numberOfRows();
   }

   @Override
   public boolean equals(Object o) {
      return o != null && o instanceof Vector && Arrays.equals(toArray(), Cast.<Vector>as(o).toArray());
   }

   @Override
   public double get(int index) {
      return entries.get(index, column);
   }

   @Override
   public int hashCode() {
      return Arrays.hashCode(toArray());
   }

   @Override
   public Vector increment(int index, double amount) {
      entries.increment(index, column, amount);
      return this;
   }

   @Override
   public boolean isDense() {
      return true;
   }

   @Override
   public boolean isSparse() {
      return false;
   }


   @Override
   public Vector set(int index, double value) {
      entries.set(index, column, value);
      return this;
   }

   @Override
   public int size() {
      return entries.numberOfRows();
   }

   @Override
   public String toString() {
      return Arrays.toString(toArray());
   }

   @Override
   public Vector zero() {
      for (int r = 0; r < entries.numberOfRows(); r++) {
         set(r, 0d);
      }
      return this;
   }
}
