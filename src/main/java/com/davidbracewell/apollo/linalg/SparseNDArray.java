package com.davidbracewell.apollo.linalg;

import com.davidbracewell.conversion.Cast;
import com.davidbracewell.guava.common.base.Preconditions;
import lombok.NonNull;
import org.apache.mahout.math.function.IntDoubleProcedure;
import org.apache.mahout.math.list.IntArrayList;
import org.apache.mahout.math.map.OpenIntDoubleHashMap;

import java.util.Arrays;
import java.util.Iterator;
import java.util.function.Consumer;

/**
 * @author David B. Bracewell
 */
public class SparseNDArray implements NDArray {

   private Shape shape;
   private OpenIntDoubleHashMap storage = new OpenIntDoubleHashMap();

   public SparseNDArray(int nRows, int nCols) {
      this.shape = Shape.shape(nRows, nCols);
   }

   public SparseNDArray(@NonNull Shape shape) {
      this.shape = shape.copy();
   }

   @Override
   public NDArray compress() {
      this.storage.trimToSize();
      return this;
   }

   @Override
   public NDArray copy() {
      SparseNDArray copy = new SparseNDArray(this.shape);
      storage.forEachPair((index, value) -> {
         copy.storage.put(index, value);
         return true;
      });
      return copy;
   }

   @Override
   public boolean equals(Object obj) {
      if (obj == null || !(obj instanceof NDArray)) {
         return false;
      }
      NDArray other = Cast.as(obj);
      return shape.dimensionMatch(other.shape()) && Arrays.equals(toArray(), other.toArray());
   }

   @Override
   public void forEachSparse(@NonNull Consumer<Entry> consumer) {
      storage.forEachPair((index, value) -> {
         consumer.accept(new EntryImpl(index));
         return true;
      });
   }

   @Override
   public double get(int dimension) {
      return this.storage.get(dimension);
   }

   @Override
   public double get(int i, int j) {
      return this.storage.get(shape.rowMajorIndex(i, j));
   }

   @Override
   public NDArrayFactory getFactory() {
      return NDArrayFactory.SparseNDArrayFactory.INSTANCE;
   }

   @Override
   public Iterator<Entry> iterator() {
      return new Alliterator();
   }


   @Override
   public double max() {
      Max max = new Max();
      storage.forEachPair(max);
      if (max.maxv == Double.NEGATIVE_INFINITY) {
         return size() == length() ? max.maxv : 0d;
      }
      return max.maxv;
   }

   @Override
   public double min() {
      Min min = new Min();
      storage.forEachPair(min);
      if (length() == size()) {
         return min.minv;
      }
      return Math.min(0, min.minv);
   }

   @Override
   public NDArray set(int index, double value) {
      Preconditions.checkArgument(index >= 0 && index < length(), "Invalid index: " + index + " (" + length() + ")");
      if (value == 0) {
         this.storage.removeKey(index);
      } else {
         this.storage.put(index, value);
      }
      return this;
   }

   @Override
   public NDArray set(@NonNull Subscript subscript, double value) {
      shape.checkSubscript(subscript);
      if (value == 0) {
         this.storage.removeKey(shape.rowMajorIndex(subscript));
      } else {
         this.storage.put(shape.rowMajorIndex(subscript), value);
      }
      return this;
   }

   @Override
   public NDArray set(int r, int c, double value) {
      return set(shape.rowMajorIndex(r, c), value);
   }

   @Override
   public Shape shape() {
      return shape;
   }

   @Override
   public int size() {
      return storage.size();
   }

   @Override
   public Iterator<Entry> sparseIterator() {
      return new SparseIterator();
   }

   @Override
   public double sum() {
      Summer summer = new Summer();
      storage.forEachPair(summer);
      return summer.sum;
   }

   @Override
   public double[][] to2DArray() {
      final double[][] array = new double[shape.i][shape.j];
//      storage.forEachPair((index, value) -> {
//         Subscript ss = Subscript.fromIndex(index, shape.i);
//         array[ss.i][ss.j] = value;
//         return true;
//      });
      return array;
   }

   @Override
   public double[] toArray() {
      final double[] array = new double[length()];
      storage.forEachPair((index, value) -> {
         array[index] = value;
         return true;
      });
      return array;
   }

   @Override
   public String toString() {
      return Arrays.toString(toArray());
   }

   static class Max implements IntDoubleProcedure {
      double maxv = Double.NEGATIVE_INFINITY;

      @Override
      public boolean apply(int i, double v) {
         if (v > maxv) {
            maxv = v;
         }
         return true;
      }
   }

   static class Min implements IntDoubleProcedure {
      double minv = Double.POSITIVE_INFINITY;

      @Override
      public boolean apply(int i, double v) {
         if (v < minv) {
            minv = v;
         }
         return true;
      }
   }

   static class Summer implements IntDoubleProcedure {
      double sum = 0;

      @Override
      public boolean apply(int i, double v) {
         sum += v;
         return true;
      }
   }

   private class EntryImpl implements NDArray.Entry {
      private static final long serialVersionUID = 1L;
      final int i;
      final int j;
      final int index;

      private EntryImpl(int index) {
         this.index = index;
         Subscript ss = shape.fromRowMajorIndex(index);
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

      public SparseIterator() {
         keyList = storage.keys();
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
}// END OF SparseNDArray
