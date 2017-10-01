package com.davidbracewell.apollo.linalg;

import com.davidbracewell.guava.common.base.Preconditions;

import java.util.Collections;
import java.util.Iterator;

/**
 * @author David B. Bracewell
 */
public class ScalarNDArray implements NDArray {
   private double value;

   public ScalarNDArray(double value) {
      this.value = value;
   }

   @Override
   public NDArray copy() {
      return new ScalarNDArray(value);
   }

   @Override
   public boolean equals(Object obj) {
      return obj != null && obj instanceof ScalarNDArray && ((ScalarNDArray) obj).value == value;
   }

   @Override
   public double get(int index) {
      Preconditions.checkPositionIndex(index, 1);
      return value;
   }

   @Override
   public double get(int i, int j) {
      Preconditions.checkPositionIndex(i, 0, "Invalid row");
      Preconditions.checkPositionIndex(j, 1, "Invalid column");
      return value;
   }

   @Override
   public NDArrayFactory getFactory() {
      return NDArrayFactory.DENSE_DOUBLE;
   }


   @Override
   public int hashCode() {
      return Double.hashCode(value);
   }

   @Override
   public Iterator<Entry> iterator() {
      return Collections.<NDArray.Entry>singleton(new ScalarEntry()).iterator();
   }

   @Override
   public NDArray set(int index, double value) {
      Preconditions.checkPositionIndex(index, 1);
      this.value = value;
      return this;
   }

   @Override
   public NDArray set(int r, int c, double value) {
      Preconditions.checkPositionIndex(r, 0, "Invalid row");
      Preconditions.checkPositionIndex(c, 1, "Invalid column");
      this.value = value;
      return this;
   }

   @Override
   public Shape shape() {
      return new Shape(1, 1);
   }

   @Override
   public double[][] to2DArray() {
      return new double[][]{{value}};
   }

   @Override
   public double[] toArray() {
      return new double[]{value};
   }

   @Override
   public String toString() {
      return Double.toString(value);
   }

   class ScalarEntry implements NDArray.Entry {

      @Override
      public int getI() {
         return 0;
      }

      @Override
      public int getIndex() {
         return 0;
      }

      @Override
      public int getJ() {
         return 0;
      }

      @Override
      public double getValue() {
         return value;
      }

      @Override
      public void setValue(double value) {
         ScalarNDArray.this.value = value;
      }
   }
}// END OF ScalarNDArray
