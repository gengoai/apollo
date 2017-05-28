package com.davidbracewell.apollo.linalg;

import com.davidbracewell.guava.common.base.Preconditions;
import lombok.EqualsAndHashCode;

/**
 * The type Single point vector.
 *
 * @author David B. Bracewell
 */
@EqualsAndHashCode
public class SinglePointVector implements Vector {
   private double value;

   /**
    * Instantiates a new Single point vector.
    */
   public SinglePointVector() {
      this(0);
   }

   /**
    * Instantiates a new Single point vector.
    *
    * @param value the value
    */
   public SinglePointVector(double value) {
      this.value = value;
   }

   @Override
   public Vector compress() {
      return this;
   }

   @Override
   public Vector copy() {
      return new SinglePointVector(value);
   }

   @Override
   public int dimension() {
      return 1;
   }

   @Override
   public double get(int index) {
      Preconditions.checkPositionIndex(index, 1);
      return value;
   }

   @Override
   public Vector increment(int index, double amount) {
      Preconditions.checkPositionIndex(index, 1);
      value += amount;
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
      Preconditions.checkArgument(newDimension >= 0, "dimension must be >= 0");
      return new DenseVector(newDimension).set(0, value);
   }

   @Override
   public Vector set(int index, double value) {
      Preconditions.checkPositionIndex(index, 1);
      this.value = value;
      return this;
   }

   @Override
   public int size() {
      return 1;
   }

   @Override
   public Vector slice(int from, int to) {
      Preconditions.checkPositionIndex(from, 1);
      Preconditions.checkPositionIndex(to, 1);
      if (to - from == 1) {
         return new SparseVector(0);
      }
      return copy();
   }

   @Override
   public double[] toArray() {
      return new double[]{value};
   }

   @Override
   public String toString() {
      return "[" + value + "]";
   }

   @Override
   public Vector zero() {
      return new SinglePointVector(0d);
   }
}// END OF SinglePointVector
