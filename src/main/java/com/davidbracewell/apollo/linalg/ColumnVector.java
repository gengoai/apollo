package com.davidbracewell.apollo.linalg;

import com.davidbracewell.conversion.Cast;
import com.davidbracewell.guava.common.base.Preconditions;

import java.io.Serializable;
import java.util.Arrays;

/**
 * @author David B. Bracewell
 */
class ColumnVector implements Vector, Serializable {
   private static final long serialVersionUID = 1L;
   private Matrix entries;
   final int column;

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
   public Vector redim(int newDimension) {
      Vector v = new DenseVector(newDimension);
      for (int r = 0; r < Math.min(entries.numberOfRows(), newDimension); r++) {
         v.set(r, get(r));
      }
      return v;
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
   public Vector slice(int from, int to) {
      Preconditions.checkArgument(from < to, from + " must be less than " + to);
      Preconditions.checkElementIndex(to, entries.numberOfRows());
      Vector v = new DenseVector(to - from);
      for (int r = from; r < to; r++) {
         v.set(r, get(r));
      }
      return v;
   }

   @Override
   public double[] toArray() {
      double[] array = new double[entries.numberOfRows()];
      for (int r = 0; r < entries.numberOfRows(); r++) {
         array[r] = get(r);
      }
      return array;
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
