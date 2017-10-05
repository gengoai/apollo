package com.davidbracewell.apollo.linear;

import java.util.Collections;
import java.util.Iterator;

/**
 * @author David B. Bracewell
 */
public class EmptyNDArray extends NDArray {

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
   public NDArray set(int index, double value) {
      throw new IndexOutOfBoundsException();
   }

   @Override
   public NDArray set(int r, int c, double value) {
      throw new IndexOutOfBoundsException();
   }

   @Override
   public Shape shape() {
      return Shape.shape(0, 0);
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
