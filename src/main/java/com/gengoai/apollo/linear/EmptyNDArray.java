package com.gengoai.apollo.linear;

import java.util.Collections;
import java.util.Iterator;

/**
 * An NDArray that has no rows or columns, i.e. is empty
 *
 * @author David B. Bracewell
 */
public final class EmptyNDArray extends NDArray {

   public static final NDArray INSTANCE = new EmptyNDArray();

   private EmptyNDArray() {

   }


   @Override
   public NDArray copyData() {
      return new EmptyNDArray();
   }

   @Override
   public boolean equals(Object o) {
      return o != null && o instanceof EmptyNDArray;
   }

   @Override
   public double get(int index) {
      throw new IndexOutOfBoundsException();
   }

   @Override
   public double get(int i, int j) {
      throw new IndexOutOfBoundsException();
   }

   @Override
   public NDArrayFactory getFactory() {
      return NDArrayFactory.DENSE_DOUBLE;
   }

   @Override
   public NDArray reshape(int numRows, int numCols) {
      throw new IllegalStateException();
   }

   @Override
   public int hashCode() {
      return getClass().hashCode();
   }

   @Override
   public boolean isSparse() {
      return false;
   }

   @Override
   public Iterator<Entry> iterator() {
      return Collections.emptyIterator();
   }

   @Override
   public NDArray mmul(NDArray other) {
      checkDimensionMatch(numCols(), numRows());
      return new EmptyNDArray();
   }

   @Override
   public int numCols() {
      return 0;
   }

   @Override
   public int numRows() {
      return 0;
   }

   @Override
   public NDArray set(int index, double value) {
      throw new IndexOutOfBoundsException();
   }

   @Override
   public NDArray set(int r, int c, double value) {
      throw new IndexOutOfBoundsException();
   }

   @Override
   public int size() {
      return 0;
   }

   @Override
   public Iterator<Entry> sparseColumnIterator(int column) {
      return Collections.emptyIterator();
   }

   @Override
   public Iterator<Entry> sparseRowIterator(int row) {
      return Collections.emptyIterator();
   }

   @Override
   public double[][] to2DArray() {
      return new double[0][];
   }

   @Override
   public double[] toArray() {
      return new double[0];
   }

   @Override
   public String toString() {
      return "[]";
   }

}// END OF EmptyNDArray
