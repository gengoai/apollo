package com.davidbracewell.apollo.linear.sparse;

import com.davidbracewell.apollo.linear.NDArray;
import com.davidbracewell.apollo.linear.NDArrayFactory;
import org.apache.mahout.math.list.IntArrayList;
import org.apache.mahout.math.map.OpenIntDoubleHashMap;

import java.util.Iterator;

/**
 * @author David B. Bracewell
 */
public class SparseVector extends NDArray {
   private int nRows;
   private int nCols;
   private final OpenIntDoubleHashMap storage = new OpenIntDoubleHashMap();

   public SparseVector(int nRows, int nCols) {
      this.nRows = nRows;
      this.nCols = nCols;
   }

   @Override
   protected NDArray copyData() {
      SparseVector sv = new SparseVector(nRows, nCols);
      storage.forEachPair((i, v) -> {
         sv.storage.put(i, v);
         return true;
      });
      return sv;
   }

   @Override
   public double get(int index) {
      return storage.get(index);
   }

   @Override
   public double get(int i, int j) {
      return storage.get(toIndex(i, j));
   }

   @Override
   public NDArrayFactory getFactory() {
      return NDArrayFactory.SPARSE_VECTOR;
   }

   @Override
   public NDArray reshape(int numRows, int numCols) {
      this.nCols = numCols;
      this.nRows = numRows;
      return this;
   }

   @Override
   public boolean isSparse() {
      return true;
   }

   @Override
   public Iterator<Entry> iterator() {
      return new Iterator<Entry>() {
         int i = 0;

         @Override
         public boolean hasNext() {
            return i < length();
         }

         @Override
         public Entry next() {
            Entry e = new EntryImpl(i);
            i++;
            return e;
         }
      };
   }

   @Override
   public int size() {
      return storage.size();
   }

   @Override
   public Iterator<Entry> sparseIterator() {
      return new Iterator<Entry>() {
         int i = 0;
         IntArrayList keys = storage.keys();

         @Override
         public boolean hasNext() {
            return i < keys.size();
         }

         @Override
         public Entry next() {
            Entry e = new EntryImpl(keys.get(i));
            i++;
            return e;
         }
      };
   }

   @Override
   public Iterator<Entry> sparseOrderedIterator() {
      final IntArrayList keys = storage.keys();
      keys.sort();
      return new Iterator<Entry>() {
         int i = 0;

         @Override
         public boolean hasNext() {
            return i < keys.size();
         }

         @Override
         public Entry next() {
            Entry e = new EntryImpl(keys.get(i));
            i++;
            return e;
         }
      };
   }

   private class EntryImpl implements NDArray.Entry {
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
         return storage.get(index);
      }

      @Override
      public void setValue(double value) {
         storage.put(index, value);
      }
   }

   @Override
   public NDArray mmul(NDArray other) {
      NDArray toReturn = getFactory().zeros(numRows(), other.numCols());
      storage.forEachPair((index, value) -> {
         int row = toRow(index);
         int column = toColumn(index);
         other.sparseRowIterator(column)
              .forEachRemaining(e2 -> toReturn.increment(row, e2.getJ(), e2.getValue() * value));
         return true;
      });
      return toReturn;
   }

   @Override
   public NDArray compress() {
      storage.trimToSize();
      return this;
   }

   @Override
   public int numCols() {
      return nCols;
   }

   @Override
   public int numRows() {
      return nRows;
   }

   @Override
   public NDArray set(int index, double value) {
      storage.put(index, value);
      return this;
   }

   @Override
   public NDArray set(int r, int c, double value) {
      storage.put(toIndex(r, c), value);
      return this;
   }

   @Override
   public Iterator<Entry> sparseColumnIterator(int column) {
      return new Iterator<Entry>() {
         int i = 0;
         Entry e = null;

         private boolean advance() {
            if (e != null) {
               return true;
            }
            while (i < nRows && !storage.containsKey(toIndex(i, column))) {
               i++;
            }
            if (i < nRows) {
               e = new EntryImpl(toIndex(i, column));
               return true;
            }
            return false;
         }

         @Override
         public boolean hasNext() {
            return advance();
         }

         @Override
         public Entry next() {
            advance();
            Entry toR = e;
            i++;
            e = null;
            return toR;
         }
      };
   }

   @Override
   public Iterator<Entry> sparseRowIterator(int row) {
      return new Iterator<Entry>() {
         int i = 0;
         Entry e = null;

         private boolean advance() {
            if (e != null) {
               return true;
            }
            while (i < nCols && !storage.containsKey(toIndex(row, i))) {
               i++;
            }
            if (i < nCols) {
               e = new EntryImpl(toIndex(row, i));
               return true;
            }
            return false;
         }

         @Override
         public boolean hasNext() {
            return advance();
         }

         @Override
         public Entry next() {
            advance();
            Entry toR = e;
            i++;
            e = null;
            return toR;
         }
      };
   }


}//END OF SparseVector
