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

import com.davidbracewell.guava.common.base.Preconditions;
import it.unimi.dsi.fastutil.ints.Int2DoubleOpenHashMap;
import lombok.NonNull;

import java.util.*;
import java.util.stream.IntStream;

/**
 * <p>Sparse implementation of Matrix stored in row-major order.</p>
 *
 * @author David B. Bracewell
 */
public class SparseMatrix extends AbstractMatrix {
   private static final long serialVersionUID = -3802597548916836308L;
   final private int numberOfRows;
   final private int colDimension;
   private volatile Int2DoubleOpenHashMap matrix;

   /**
    * Instantiates a new Sparse matrix.
    *
    * @param numRows    the number of rows
    * @param numColumns the number of columns
    */
   public SparseMatrix(int numRows, int numColumns) {
      this.colDimension = numColumns;
      this.numberOfRows = numRows;
      this.matrix = new Int2DoubleOpenHashMap();
   }

   /**
    * Instantiates a new Sparse matrix.
    *
    * @param matrix the matrix to copy
    */
   public SparseMatrix(@NonNull Matrix matrix) {
      this(matrix.numberOfRows(), matrix.numberOfColumns());
      matrix.forEachSparse(e -> set(e.row, e.column, e.value));
   }

   /**
    * Instantiates a new Sparse matrix from a number of row vectors.
    *
    * @param vectors the vectors
    */
   public SparseMatrix(@NonNull Vector... vectors) {
      this(Arrays.asList(vectors));
   }

   /**
    * Instantiates a new Sparse matrix from a number of row vectors.
    *
    * @param vectors the vectors
    */
   public SparseMatrix(@NonNull List<Vector> vectors) {
      if (vectors.size() == 0) {
         this.colDimension = 0;
         this.numberOfRows = 0;
      } else {
         this.colDimension = vectors.get(0).dimension();
         this.numberOfRows = vectors.size();
      }
      this.matrix = new Int2DoubleOpenHashMap();
      for (int i = 0; i < vectors.size(); i++) {
         setRow(i, vectors.get(i));
      }
   }

   /**
    * Zeroes matrix.
    *
    * @param numberOfRows    the number of rows
    * @param numberOfColumns the number of columns
    * @return the matrix
    */
   public static Matrix zeroes(int numberOfRows, int numberOfColumns) {
      return new SparseMatrix(numberOfRows, numberOfColumns);
   }

   /**
    * Ones matrix.
    *
    * @param numberOfRows    the number of rows
    * @param numberOfColumns the number of columns
    * @return the matrix
    */
   public static Matrix ones(int numberOfRows, int numberOfColumns) {
      return new SparseMatrix(numberOfRows, numberOfColumns).incrementSelf(1);
   }

   /**
    * Unit matrix.
    *
    * @param size the size
    * @return the matrix
    */
   public static Matrix unit(int size) {
      Matrix m = new SparseMatrix(size, size);
      for (int r = 0; r < size; r++) {
         m.set(r, r, 1d);
      }
      return m;
   }

   /**
    * Random matrix.
    *
    * @param numberOfRows    the number of rows
    * @param numberOfColumns the number of columns
    * @return the matrix
    */
   public static Matrix random(int numberOfRows, int numberOfColumns) {
      Matrix m = new SparseMatrix(numberOfRows, numberOfColumns);
      IntStream.range(0, numberOfRows)
               .parallel()
               .forEach(r -> {
                  for (int c = 0; c < numberOfColumns; c++) {
                     m.set(r, c, Math.random());
                  }
               });
      return m;
   }


   @Override
   protected Matrix createNew(int nRows, int nCols) {
      return new SparseMatrix(nRows, nCols);
   }

   @Override
   public Iterator<Entry> nonZeroIterator() {
      return new Iterator<Entry>() {
         private PrimitiveIterator.OfInt rowItr = IntStream.range(0, numberOfRows).iterator();
         private int row;
         private Integer currentColumn = null;
         private Iterator<Vector.Entry> colItr;

         private boolean advance() {

            while (currentColumn == null) {
               if (colItr == null && !rowItr.hasNext()) {
                  return false;
               } else if (colItr == null) {
                  row = rowItr.next();
                  colItr = row(row).nonZeroIterator();
               } else if (colItr.hasNext()) {
                  currentColumn = colItr.next().getIndex();
               } else {
                  currentColumn = null;
                  colItr = null;
               }
            }

            return currentColumn != null;
         }

         @Override
         public boolean hasNext() {
            return advance();
         }

         @Override
         public Entry next() {
            if (!advance()) {
               throw new NoSuchElementException();
            }
            int c = currentColumn;
            currentColumn = null;
            return new Matrix.Entry(row, c, get(row, c));
         }
      };
   }

   @Override
   public Iterator<Entry> orderedNonZeroIterator() {
      return new Iterator<Entry>() {
         private PrimitiveIterator.OfInt rowItr = IntStream.range(0, numberOfRows).iterator();
         private int row;
         private Integer currentColumn = null;
         private Iterator<Vector.Entry> colItr;

         private boolean advance() {

            while (currentColumn == null) {
               if (colItr == null && !rowItr.hasNext()) {
                  return false;
               } else if (colItr == null) {
                  row = rowItr.next();
                  colItr = row(row).orderedNonZeroIterator();
               } else if (colItr.hasNext()) {
                  currentColumn = colItr.next().getIndex();
               } else {
                  currentColumn = null;
                  colItr = null;
               }
            }

            return currentColumn != null;
         }

         @Override
         public boolean hasNext() {
            return advance();
         }

         @Override
         public Entry next() {
            if (!advance()) {
               throw new NoSuchElementException();
            }
            int c = currentColumn;
            currentColumn = null;
            return new Matrix.Entry(row, c, get(row, c));
         }
      };
   }

   @Override
   public double get(int row, int column) {
      Preconditions.checkElementIndex(row, numberOfRows());
      Preconditions.checkElementIndex(column, numberOfColumns());
      return matrix.get(encode(row, column));
   }

   @Override
   public void set(int row, int column, double value) {
      Preconditions.checkElementIndex(row, numberOfRows());
      Preconditions.checkElementIndex(column, numberOfColumns());
      synchronized (this) {
         matrix.put(encode(row, column), value);
      }
   }

   @Override
   public int numberOfRows() {
      return numberOfRows;
   }

   @Override
   public int numberOfColumns() {
      return colDimension;
   }

   @Override
   public boolean isSparse() {
      return true;
   }

   private int encode(int row, int col) {
      return (row * numberOfColumns()) + col;
   }

   @Override
   public Matrix increment(int row, int col, double amount) {
      Preconditions.checkElementIndex(row, numberOfRows());
      Preconditions.checkElementIndex(col, numberOfColumns());
      synchronized (this) {
         matrix.addTo(encode(row, col), amount);
      }
      return this;
   }

   @Override
   public String toString() {
      if (numberOfRows() > 10 || numberOfColumns() > 10) {
         return numberOfRows() + "x" + numberOfColumns();
      }
      StringBuilder builder = new StringBuilder("[ ");
      for (int row = 0; row < numberOfRows(); row++) {
         builder.append("[ ");
         for (int col = 0; col < numberOfColumns(); col++) {
            builder.append(get(row, col)).append(", ");
         }
         builder.append("] ");
      }
      return builder.append("]").toString();
   }

   @Override
   public Matrix copy() {
      return new SparseMatrix(this);
   }

}//END OF SparseMatrix
