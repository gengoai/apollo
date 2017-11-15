package com.davidbracewell.apollo.linear;

import com.davidbracewell.guava.common.base.Preconditions;
import com.davidbracewell.guava.common.collect.Iterators;

import java.util.Collections;
import java.util.Iterator;

/**
 * @author David B. Bracewell
 */
public class ScalarNDArray extends NDArray {
   private double value;

   public ScalarNDArray(double value) {
      this.value = value;
   }

   @Override
   public NDArray copyData() {
      return new ScalarNDArray(value);
   }

   @Override
   public double get(int index) {
      Preconditions.checkPositionIndex(index, 1);
      return value;
   }

   @Override
   public NDArray reshape(int numRows, int numCols) {
      throw new IllegalStateException();
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
   public boolean isSparse() {
      return false;
   }

   @Override
   public Iterator<Entry> iterator() {
      return Collections.<NDArray.Entry>singleton(new ScalarEntry()).iterator();
   }

   @Override
   public NDArray mmul(NDArray other) {
      checkDimensionMatch(numCols(), numRows());
      return new ScalarNDArray(scalarValue() * other.scalarValue());
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
   public Iterator<Entry> sparseColumnIterator(int column) {
      return Iterators.filter(iterator(), e -> e.getValue() != 0);
   }

   @Override
   public Iterator<Entry> sparseRowIterator(int row) {
      return Iterators.filter(iterator(), e -> e.getValue() != 0);
   }

   @Override
   public int numCols() {
      return 1;
   }

   @Override
   public int numRows() {
      return 1;
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
      return "[" + Double.toString(value) + "]";
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
