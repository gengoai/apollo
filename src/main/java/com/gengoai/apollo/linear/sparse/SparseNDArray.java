package com.gengoai.apollo.linear.sparse;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.conversion.Cast;
import org.apache.mahout.math.function.IntDoubleProcedure;
import org.apache.mahout.math.list.IntArrayList;

import java.util.BitSet;
import java.util.Iterator;

/**
 * @author David B. Bracewell
 */
public abstract class SparseNDArray extends NDArray {
   private static final long serialVersionUID = 1L;
   protected int nRows;
   protected int nCols;
   protected BitSet bits;

   public SparseNDArray(int nRows, int nCols) {
      this.nRows = nRows;
      bits = new BitSet(nRows * nCols);
      this.nCols = nCols;
   }

   protected abstract double adjustOrPutValue(int index, double amount);

   @Override
   protected NDArray copyData() {
      SparseNDArray ndArray = Cast.as(getFactory().zeros(nRows, nCols));
      forEachPair((index, value) -> {
         ndArray.set(index, value);
         return true;
      });
      ndArray.bits.or(bits);
      return ndArray;
   }

   protected Entry createEntry(int index) {
      return new EntryImpl(index);
   }

   @Override
   public final NDArray decrement(int index, double amount) {
      double v = adjustOrPutValue(index, -amount);
      if (v == 0) {
         removeIndex(index);
         bits.set(index, false);
      } else {
         bits.set(index, true);
      }
      return this;
   }

   protected abstract void forEachPair(IntDoubleProcedure procedure);

   @Override
   public final double get(int i, int j) {
      return get(toIndex(i, j));
   }

   @Override
   public final NDArray increment(int index, double amount) {
      double v = adjustOrPutValue(index, amount);
      if (v == 0) {
         removeIndex(index);
         bits.set(index, false);
      } else {
         bits.set(index, true);
      }
      return this;
   }

   @Override
   public final boolean isSparse() {
      return true;
   }

   @Override
   public final Iterator<Entry> iterator() {
      return new Iterator<Entry>() {
         int i = 0;

         @Override
         public boolean hasNext() {
            return i < length();
         }

         @Override
         public Entry next() {
            Entry e = createEntry(i);
            i++;
            return e;
         }
      };
   }

   @Override
   public final NDArray mmul(NDArray other) {
      NDArray toReturn = getFactory().zeros(numRows(), other.numCols());
      forEachPair((index, value) -> {
         int row = toRow(index);
         int column = toColumn(index);
         other.sparseRowIterator(column)
              .forEachRemaining(e2 -> toReturn.increment(row, e2.getJ(), e2.getValue() * value));
         return true;
      });
      return toReturn;
   }

   protected abstract IntArrayList nonZeroIndexes();

   @Override
   public final int numCols() {
      return nCols;
   }

   @Override
   public final int numRows() {
      return nRows;
   }

   protected abstract void removeIndex(int index);

   @Override
   public final NDArray reshape(int numRows, int numCols) {
      this.nCols = numCols;
      this.nRows = numRows;
      return this;
   }

   @Override
   public final NDArray set(int index, double value) {
      if (value == 0) {
         bits.set(index, false);
         removeIndex(index);
      } else {
         bits.set(index, true);
         setValue(index, value);
      }
      return this;
   }

   @Override
   public final NDArray set(int r, int c, double value) {
      return set(toIndex(r, c), value);
   }

   protected abstract void setValue(int index, double value);

   @Override
   public final Iterator<Entry> sparseColumnIterator(int column) {
      return new Iterator<Entry>() {
         int i = 0;

         private boolean advance() {
            while (i < nRows && !bits.get(toIndex(i, nCols))) {
               i++;
            }
            return i < nRows;
         }

         @Override
         public boolean hasNext() {
            return advance();
         }

         @Override
         public Entry next() {
            advance();
            int r = i;
            i++;
            return createEntry(toIndex(r, column));
         }
      };
   }

   @Override
   public final Iterator<Entry> sparseIterator() {
      return new Iterator<Entry>() {
         int i = 0;
         IntArrayList keys = nonZeroIndexes();


         @Override
         public boolean hasNext() {
            return i < keys.size();
         }

         @Override
         public Entry next() {
            Entry e = createEntry(keys.get(i));
            i++;
            return e;
         }
      };
   }

   @Override
   public final Iterator<Entry> sparseOrderedIterator() {
      final IntArrayList keys = nonZeroIndexes();
      keys.sort();
      return new Iterator<Entry>() {
         int i = 0;

         @Override
         public boolean hasNext() {
            return i < keys.size();
         }

         @Override
         public Entry next() {
            Entry e = createEntry(keys.get(i));
            i++;
            return e;
         }
      };
   }

   @Override
   public final Iterator<Entry> sparseRowIterator(int row) {
      return new Iterator<Entry>() {
         int i = 0;

         private boolean advance() {
            while (i < nCols && !bits.get(toIndex(row, i))) {
               i++;
            }
            return i < nCols;
         }

         @Override
         public boolean hasNext() {
            return advance();
         }

         @Override
         public Entry next() {
            advance();
            int c = i;
            i++;
            return createEntry(toIndex(row, c));
         }
      };
   }

   private class EntryImpl implements Entry {
      final int r;
      final int c;
      final int index;

      private EntryImpl(int index) {
         this.index = index;
         this.r = toRow(index);
         this.c = toColumn(index);
      }

      @Override
      public int getI() {
         return r;
      }

      @Override
      public int getIndex() {
         return index;
      }

      @Override
      public int getJ() {
         return c;
      }

      @Override
      public double getValue() {
         return SparseNDArray.this.get(index);
      }

      @Override
      public void setValue(double value) {
         set(index, value);
      }

      @Override
      public String toString() {
         return String.format("(%d, %d, %f)", r, c, getValue());
      }

   }

}// END OF SparseNDArray
