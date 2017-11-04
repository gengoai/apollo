package com.davidbracewell.apollo.linear.sparse;

import com.davidbracewell.Copyable;
import com.davidbracewell.apollo.linear.NDArray;
import com.davidbracewell.apollo.linear.Shape;
import lombok.Getter;
import org.apache.mahout.math.list.DoubleArrayList;
import org.apache.mahout.math.list.IntArrayList;

import java.util.Iterator;
import java.util.function.Consumer;

/**
 * @author David B. Bracewell
 */
public class CCS implements Copyable<CCS> {
   DoubleArrayList values = new DoubleArrayList();
   IntArrayList row_index = new IntArrayList();
   IntArrayList col_ptr = new IntArrayList();
   @Getter
   Shape shape;
   int cardinality = 0;


   public CCS(double[][] data) {
      this.shape = Shape.shape(data.length, data[0].length);
      for (int col = 0; col < shape.j; col++) {
         col_ptr.add(values.size());
         for (int row = 0; row < data.length; row++) {
            if (data[row][col] != 0) {
               values.add(data[row][col]);
               row_index.add(row);
            }
         }
      }
      col_ptr.add(values.size());
   }

   public CCS(int nR, int nC) {
      this(Shape.shape(nR, nC));
   }

   public CCS(Shape shape) {
      this.shape = shape;
      for (int i = 0; i <= shape.j; i++) {
         col_ptr.add(0);
      }
   }

   public static void main(String[] args) {
      CCS m = new CCS(2, 2);
      m.set(0, 1, 20);
      m.set(0, 0, 10);
      m.set(1, 0, 30);
      m.set(1, 1, 40);
      m.sparseRowIterator(0).forEachRemaining(System.out::println);
      m.sparseRowIterator(1).forEachRemaining(System.out::println);
   }

   public void clear() {
      this.cardinality = 0;
      this.values.clear();
      for (int i = 0; i < this.col_ptr.size(); i++) {
         this.col_ptr.set(i, 0);
      }
      this.row_index.clear();
   }

   @Override
   public CCS copy() {
      CCS toReturn = new CCS(shape);
      toReturn.values = values.copy();
      toReturn.row_index = row_index.copy();
      toReturn.col_ptr = col_ptr.copy();
      toReturn.cardinality = cardinality;
      return toReturn;
   }

   private int findRowIndex(int row, int lower, int upper) {
      if (upper - lower == 0 || row_index.isEmpty() || upper == 0 || row > row_index.get(upper - 1)) {
         return upper;
      }
      return row_index.binarySearchFromTo(row, lower, upper);
   }

   public void forEach(Consumer<NDArray.Entry> consumer) {
      sparseColumnMajorIterator().forEachRemaining(consumer);
   }

   public double get(int dimension) {
      int i = findRowIndex(dimension, 0, values.size());
      if (i >= 0 && i < values.size()) {
         return values.get(i);
      }
      return 0d;
   }

   public double get(int r, int c) {
      int index = findRowIndex(r, col_ptr.get(c), col_ptr.get(c + 1));
      if (index >= 0 && index < col_ptr.get(c + 1) && row_index.get(index) == r) {
         return values.get(index);
      }
      return 0d;
   }

   private void insert(int index, int row, int col, double value) {
      index = Math.abs(index);
      values.beforeInsert(index, value);
      row_index.beforeInsert(index, row);
      for (int jj = col + 1; jj < shape.j + 1; jj++) {
         col_ptr.set(jj, col_ptr.get(jj) + 1);
      }
      cardinality++;
   }

   public void set(int dimension, double value) {
      set(dimension, 0, value);
   }

   public void set(int r, int c, double value) {
      if (col_ptr.isEmpty()) {
         insert(0, r, c, value);
      } else {
         int index = findRowIndex(r, col_ptr.get(c), col_ptr.get(c + 1));
         if (index >= 0 && index < col_ptr.get(c + 1) && row_index.get(index) == c) {
            values.set(index, 0d);
         } else {
            insert(index, r, c, value);
         }
      }
   }

   public int size() {
      return values.size();
   }

   public Iterator<NDArray.Entry> sparseColumnIterator(int column) {
      return new Iterator<NDArray.Entry>() {
         int upper = col_ptr.get(column + 1);
         int index = col_ptr.get(column);

         @Override
         public boolean hasNext() {
            return index < upper && index < values.size();
         }

         @Override
         public NDArray.Entry next() {
            E out = new E(row_index.get(index), column, index);
            index++;
            return out;
         }
      };
   }

   public Iterator<NDArray.Entry> sparseColumnMajorIterator() {
      return new Iterator<NDArray.Entry>() {
         int col = 0;
         Iterator<NDArray.Entry> itr = null;

         private boolean advance() {
            if (itr != null && itr.hasNext()) {
               return true;
            }
            if (col >= shape.j) {
               return false;
            }
            while (itr == null || !itr.hasNext()) {
               if (col >= shape.j) {
                  return false;
               }
               itr = sparseColumnIterator(col);
               col++;
            }
            return itr != null && itr.hasNext();
         }

         @Override
         public boolean hasNext() {
            return advance();
         }

         @Override
         public NDArray.Entry next() {
            advance();
            return itr.next();
         }
      };
   }

   public Iterator<NDArray.Entry> sparseRowIterator(int row) {
      return new Iterator<NDArray.Entry>() {
         int col = -1;
         int index = -1;

         private boolean advance() {
            if (index >= 0) {
               return true;
            }
            while (index < 0 && col < shape.j - 1) {
               col++;
               index = findRowIndex(row, col_ptr.get(col), col_ptr.get(col + 1));
            }
            return index >= 0 && index < values.size();
         }

         @Override
         public boolean hasNext() {
            return advance();
         }

         @Override
         public NDArray.Entry next() {
            advance();
            E out = new E(row, col, index);
            index = -1;
            return out;
         }
      };
   }

   public double[] toArray() {
      return values.toArray(new double[values.size()]);
   }

   public class E implements NDArray.Entry {
      private final int i;
      private final int j;
      private final int index;
      private int ai;

      public E(int i, int j, int ai) {
         this.i = i;
         this.j = j;
         this.ai = ai;
         this.index = shape.colMajorIndex(i, j);
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
         return values.get(ai);
      }

      @Override
      public void setValue(double value) {
         values.set(ai, value);
      }

      @Override
      public String toString() {
         return "(" + i + ", " + j + ", " + getValue() + ")";
      }
   }


}// END OF CCS
