package com.gengoai.apollo.linear;


import com.gengoai.Validation;
import com.google.common.collect.Iterators;

import java.util.Collections;
import java.util.Iterator;

/**
 * 1x1 NDArray implementation
 *
 * @author David B. Bracewell
 */
public final class ScalarNDArray extends NDArray {
   private static final long serialVersionUID = 1L;
   private double value;

   /**
    * Instantiates a new Scalar NDArray.
    *
    * @param value the value
    */
   public ScalarNDArray(double value) {
      this.value = value;
   }

   @Override
   public NDArray copyData() {
      return new ScalarNDArray(value);
   }

   @Override
   public double get(int index) {
      Validation.checkElementIndex(index, 1);
      return value;
   }

   @Override
   public double get(int i, int j) {
      Validation.checkElementIndex(i, 0, "Invalid row");
      Validation.checkElementIndex(j, 1, "Invalid column");
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
   public int numCols() {
      return 1;
   }

   @Override
   public int numRows() {
      return 1;
   }

   @Override
   public NDArray reshape(int numRows, int numCols) {
      throw new IllegalStateException();
   }

   @Override
   public NDArray set(int index, double value) {
      Validation.checkElementIndex(index, 1);
      this.value = value;
      return this;
   }

   @Override
   public NDArray set(int r, int c, double value) {
      Validation.checkElementIndex(r, 0, "Invalid row");
      Validation.checkElementIndex(c, 1, "Invalid column");
      this.value = value;
      return this;
   }

   @Override
   public int size() {
      return 1;
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

   /**
    * The type Scalar entry.
    */
   private class ScalarEntry implements NDArray.Entry {

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
