/*
 * (c) 2005 David B. Bracewell
 *
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.davidbracewell.apollo.linalg;

import com.davidbracewell.collection.Collect;
import com.google.common.base.Preconditions;
import lombok.NonNull;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

/**
 * The type Abstract matrix.
 *
 * @author David B. Bracewell
 */
public abstract class AbstractMatrix implements Matrix, Serializable {
   private static final long serialVersionUID = 1L;

   /**
    * Create new matrix.
    *
    * @param nRows the n rows
    * @param nCols the n cols
    * @return the matrix
    */
   protected abstract Matrix createNew(int nRows, int nCols);

   @Override
   public DenseMatrix toDense() {
      return new DenseMatrix(this);
   }

   @Override
   public Matrix diag() {
      Matrix m = createNew(numberOfRows(), numberOfColumns());
      for (int r = 0; r < numberOfRows() && r < numberOfColumns(); r++) {
         m.set(r, r, get(r, r));
      }
      return m;
   }

   @Override
   public Vector column(int column) {
      Preconditions.checkElementIndex(column, numberOfColumns());
      return new ColumnVector(column);
   }

   @Override
   public Vector row(int row) {
      Preconditions.checkElementIndex(row, numberOfRows());
      return new RowVector(row);
   }

   @Override
   public void setColumn(int column, @NonNull Vector vector) {
      Preconditions.checkElementIndex(column, numberOfColumns());
      Preconditions.checkArgument(vector.dimension() == numberOfRows(), "Dimension Mismatch");
      vector.forEach(e -> set(e.index, column, e.value));
   }

   @Override
   public void setRow(int row, Vector vector) {
      Preconditions.checkElementIndex(row, numberOfRows());
      Preconditions.checkArgument(vector.dimension() == numberOfColumns(), "Dimension Mismatch");
      vector.forEach(e -> set(row, e.index, e.value));
   }

   @Override
   public Matrix addSelf(@NonNull Matrix other) {
      Preconditions.checkArgument(
         other.numberOfColumns() == numberOfColumns() && other.numberOfRows() == numberOfRows(),
         "Dimension Mismatch");
      other.forEachSparse(e -> increment(e.row, e.column, e.value));
      return this;
   }

   @Override
   public Matrix subtractSelf(@NonNull Matrix other) {
      Preconditions.checkArgument(
         other.numberOfColumns() == numberOfColumns() && other.numberOfRows() == numberOfRows(),
         "Dimension Mismatch");
      other.forEachSparse(e -> decrement(e.row, e.column, e.value));
      return this;
   }

   @Override
   public Matrix scaleSelf(@NonNull Matrix other) {
      Preconditions.checkArgument(
         other.numberOfColumns() == numberOfColumns() && other.numberOfRows() == numberOfRows(),
         "Dimension Mismatch");
      other.forEachSparse(e -> scale(e.row, e.column, e.value));
      return this;
   }

   @Override
   public Matrix multiplyVectorRowSelf(@NonNull Vector other) {
      Preconditions.checkArgument(other.dimension() == numberOfColumns(), "Dimension Mismatch");
      for (int r = 0; r < numberOfRows(); r++) {
         for (Vector.Entry e : Collect.asIterable(other.nonZeroIterator())) {
            increment(r, e.index, e.value);
         }
      }
      return this;
   }

   @Override
   public Matrix multiplyVectorColumnSelf(@NonNull Vector other) {
      Preconditions.checkArgument(other.dimension() == numberOfRows(), "Dimension Mismatch");
      forEachColumn(c -> c.multiplySelf(other));
      return this;
   }

   @Override
   public Matrix scaleSelf(double value) {
      forEachSparse(e -> set(e.row, e.column, e.value * value));
      return this;
   }

   @Override
   public Matrix incrementSelf(double value) {
      forEachRow(row -> row.mapAddSelf(value));
      return this;
   }

   @Override
   public Iterator<Vector> columnIterator() {
      return new Iterator<Vector>() {
         private AtomicInteger c = new AtomicInteger(0);

         @Override
         public boolean hasNext() {
            return c.get() < numberOfColumns();
         }

         @Override
         public Vector next() {
            if (!hasNext()) {
               throw new NoSuchElementException();
            }
            return column(c.getAndIncrement());
         }
      };
   }

   @Override
   public Iterator<Vector> rowIterator() {
      return new Iterator<Vector>() {
         private AtomicInteger r = new AtomicInteger(0);

         @Override
         public boolean hasNext() {
            return r.get() < numberOfRows();
         }

         @Override
         public Vector next() {
            if (!hasNext()) {
               throw new NoSuchElementException();
            }
            return row(r.getAndIncrement());
         }
      };
   }

   @Override
   public Matrix multiply(@NonNull Matrix m) {
      Preconditions.checkArgument(numberOfColumns() == m.numberOfRows(), "Dimension Mismatch");
      Matrix mprime = createNew(numberOfRows(), m.numberOfColumns());
      IntStream.range(0, numberOfRows()).parallel().forEach(r -> {
         for (int c = 0; c < m.numberOfColumns(); c++) {
            for (int k = 0; k < numberOfColumns(); k++) {
               mprime.increment(r, c, get(r, k) * m.get(k, c));
            }
         }
      });
      return mprime;
   }

   @Override
   public Matrix transpose() {
      Matrix T = createNew(numberOfColumns(), numberOfRows());
      forEachSparse(e -> T.set(e.column, e.row, e.value));
      return T;
   }

   @Override
   public double[][] toArray() {
      double[][] array = new double[numberOfRows()][numberOfColumns()];
      forEachSparse(e -> array[e.row][e.column] = e.value);
      return array;
   }

   /**
    * The type Column vector.
    */
   class ColumnVector implements Vector, Serializable {
      private static final long serialVersionUID = 1L;
      /**
       * The Column.
       */
      final int column;

      /**
       * Instantiates a new Column vector.
       *
       * @param column the column
       */
      ColumnVector(int column) {
         this.column = column;
      }

      @Override
      public Vector compress() {
         return this;
      }

      @Override
      public Vector copy() {
         return new DenseVector(toArray());
      }

      @Override
      public int dimension() {
         return numberOfRows();
      }

      @Override
      public double get(int index) {
         return AbstractMatrix.this.get(index, column);
      }

      @Override
      public Vector increment(int index, double amount) {
         AbstractMatrix.this.increment(index, column, amount);
         return this;
      }

      @Override
      public boolean isDense() {
         return true;
      }

      @Override
      public boolean isSparse() {
         return false;
      }

      @Override
      public Vector set(int index, double value) {
         AbstractMatrix.this.set(index, column, value);
         return this;
      }

      @Override
      public int size() {
         return numberOfRows();
      }

      @Override
      public Vector slice(int from, int to) {
         Preconditions.checkArgument(from - to > 0);
         Preconditions.checkElementIndex(to, numberOfRows());
         Vector v = new DenseVector(to - from);
         for (int r = from; r < to; r++) {
            v.set(r, get(r));
         }
         return v;
      }

      @Override
      public double[] toArray() {
         double[] array = new double[numberOfRows()];
         for (int r = 0; r < numberOfRows(); r++) {
            array[r] = get(r);
         }
         return array;
      }

      @Override
      public Vector zero() {
         for (int r = 0; r < numberOfRows(); r++) {
            set(r, 0d);
         }
         return this;
      }

      @Override
      public Vector redim(int newDimension) {
         Vector v = new DenseVector(newDimension);
         for (int r = 0; r < Math.min(numberOfRows(), newDimension); r++) {
            v.set(r, get(r));
         }
         return v;
      }

      @Override
      public String toString() {
         return Arrays.toString(toArray());
      }
   }

   /**
    * The type Row vector.
    */
   class RowVector implements Vector, Serializable {
      private static final long serialVersionUID = 1L;
      /**
       * The Row.
       */
      final int row;

      /**
       * Instantiates a new Row vector.
       *
       * @param row the row
       */
      RowVector(int row) {
         this.row = row;
      }

      @Override
      public String toString() {
         return Arrays.toString(toArray());
      }

      @Override
      public Vector compress() {
         return this;
      }

      @Override
      public Vector copy() {
         return new DenseVector(toArray());
      }

      @Override
      public int dimension() {
         return numberOfColumns();
      }

      @Override
      public double get(int index) {
         return AbstractMatrix.this.get(row, index);
      }

      @Override
      public Vector increment(int index, double amount) {
         AbstractMatrix.this.increment(row, index, amount);
         return this;
      }

      @Override
      public boolean isDense() {
         return true;
      }

      @Override
      public boolean isSparse() {
         return false;
      }

      @Override
      public Vector set(int index, double value) {
         AbstractMatrix.this.set(row, index, value);
         return this;
      }

      @Override
      public int size() {
         return numberOfColumns();
      }

      @Override
      public Vector slice(int from, int to) {
         Preconditions.checkArgument(from - to > 0);
         Preconditions.checkElementIndex(to, numberOfColumns());
         Vector v = new DenseVector(to - from);
         for (int r = from; r < to; r++) {
            v.set(r, get(r));
         }
         return v;
      }

      @Override
      public double[] toArray() {
         double[] array = new double[numberOfColumns()];
         for (int r = 0; r < numberOfColumns(); r++) {
            array[r] = get(r);
         }
         return array;
      }

      @Override
      public Vector zero() {
         for (int r = 0; r < numberOfColumns(); r++) {
            set(r, 0d);
         }
         return this;
      }

      @Override
      public Vector redim(int newDimension) {
         Vector v = new DenseVector(newDimension);
         for (int r = 0; r < Math.min(numberOfColumns(), newDimension); r++) {
            v.set(r, get(r));
         }
         return v;
      }
   }
}//END OF AbstractMatrix
