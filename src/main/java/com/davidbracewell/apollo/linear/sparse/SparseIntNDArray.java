package com.davidbracewell.apollo.linear.sparse;


import com.davidbracewell.apollo.linear.*;
import com.davidbracewell.collection.Streams;
import com.davidbracewell.guava.common.base.Preconditions;
import lombok.NonNull;
import org.apache.mahout.math.list.IntArrayList;
import org.apache.mahout.math.map.OpenIntIntHashMap;

import java.util.Iterator;
import java.util.Objects;

/**
 * @author David B. Bracewell
 */
public class SparseIntNDArray extends NDArray {
   private static final long serialVersionUID = 1L;
   private OpenIntIntHashMap storage;
   private Shape shape;

   public SparseIntNDArray(int i, int j) {
      this.storage = new OpenIntIntHashMap();
      this.shape = new Shape(i, j);
   }

   @Override
   public NDArray copyData() {
      SparseIntNDArray toReturn = new SparseIntNDArray(shape.i, shape.j);
      storage.forEachPair((index, value) -> {
         toReturn.set(index, value);
         return true;
      });
      return toReturn;
   }

   @Override
   public double get(int index) {
      Preconditions.checkPositionIndex(index, shape.length(), "Invalid index");
      return storage.get(index);
   }

   @Override
   public double get(int i, int j) {
      return get(shape.colMajorIndex(i, j));
   }

   @Override
   public NDArrayFactory getFactory() {
      return NDArrayFactory.SPARSE_INT;
   }

   @Override
   public int hashCode() {
      return Objects.hashCode(storage);
   }

   @Override
   public boolean isSparse() {
      return true;
   }

   @Override
   public Iterator<Entry> iterator() {
      return new Alliterator();
   }

   @Override
   public NDArray set(int index, double value) {
      Preconditions.checkPositionIndex(index, shape.length(), "Invalid index");
      storage.put(index, (int) value);
      return this;
   }

   @Override
   public NDArray set(int r, int c, double value) {
      return set(shape.colMajorIndex(r, c), value);
   }

   @Override
   public Shape shape() {
      return shape;
   }

   @Override
   public Iterator<Entry> sparseIterator() {
      return new SparseIterator(false);
   }

   @Override
   public Iterator<Entry> sparseOrderedIterator() {
      return new SparseIterator(true);
   }

   private class EntryImpl implements NDArray.Entry {
      private static final long serialVersionUID = 1L;
      final int i;
      final int j;
      final int index;

      private EntryImpl(int index) {
         this.index = index;
         Subscript ss = shape.fromColMajorIndex(index);
         this.i = ss.i;
         this.j = ss.j;
      }

      @Override
      public int get(@NonNull Axis axis) {
         return axis == Axis.ROW ? i : j;
      }

      @Override
      public int getI() {
         return i;
      }

      @Override
      public int getIndex() {
         return index;
      }

      @Override
      public int getJ() {
         return j;
      }

      @Override
      public double getValue() {
         return storage.get(index);
      }

      @Override
      public void setValue(double value) {
         set(index, value);
      }
   }

   private class Alliterator implements Iterator<NDArray.Entry> {
      private int index = 0;

      @Override
      public boolean hasNext() {
         return index < length();
      }

      @Override
      public Entry next() {
         Preconditions.checkElementIndex(index, length());
         return new EntryImpl(index++);
      }
   }

   private class SparseIterator implements Iterator<NDArray.Entry> {
      final IntArrayList keyList;
      int index = 0;

      public SparseIterator(boolean ordered) {
         if (ordered && isVector()) {
            keyList = storage.keys();
            keyList.quickSort();
         } else if (ordered) {
            keyList = new IntArrayList(Streams.asStream(sparseIterator())
                                              .map(e -> Subscript.from(e.getI(), e.getJ()))
                                              .sorted()
                                              .mapToInt(ss -> shape.colMajorIndex(ss))
                                              .toArray());
         } else {
            keyList = storage.keys();
         }
      }

      @Override
      public boolean hasNext() {
         return index < keyList.size();
      }

      @Override
      public Entry next() {
         Preconditions.checkElementIndex(index, keyList.size());
         int dimi = keyList.get(index);
         index++;
         return new EntryImpl(dimi);
      }
   }
}// END OF SparseIntNDArray
