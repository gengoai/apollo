package com.davidbracewell.apollo.linalg;

import com.davidbracewell.conversion.Cast;

import java.util.Collections;
import java.util.Iterator;

/**
 * @author David B. Bracewell
 */
public class EmptyNDArray implements NDArray {
   private Object label;

   @Override
   public NDArray copy() {
      NDArray toReturn = new EmptyNDArray();
      toReturn.setLabel(label);
      return toReturn;
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
   public <T> T getLabel() {
      return Cast.as(label);
   }

   @Override
   public int hashCode() {
      return getClass().hashCode();
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
   public NDArray setLabel(Object label) {
      this.label = label;
      return this;
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
