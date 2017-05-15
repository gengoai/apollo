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


import com.davidbracewell.Copyable;
import com.davidbracewell.collection.Streams;
import com.davidbracewell.guava.common.base.Preconditions;
import com.davidbracewell.tuple.Tuple2;
import lombok.NonNull;
import lombok.Value;

import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.PrimitiveIterator;
import java.util.function.Consumer;
import java.util.function.DoubleUnaryOperator;
import java.util.stream.IntStream;

import static com.davidbracewell.tuple.Tuples.$;

/**
 * <p>Interface for Matrices</p>
 *
 * @author David B. Bracewell
 */
public interface Matrix extends Copyable<Matrix>, Iterable<Matrix.Entry> {


   /**
    * The shape of the matrix as <code>number of rows</code>, <code>number of columns</code> tuple
    *
    * @return the shape
    */
   default Tuple2<Integer, Integer> shape() {
      return $(numberOfRows(), numberOfColumns());
   }

   @Override
   default Iterator<Entry> iterator() {
      return new Iterator<Entry>() {
         private PrimitiveIterator.OfInt rowItr = IntStream.range(0, numberOfRows()).iterator();
         private int row;
         private PrimitiveIterator.OfInt colItr = null;

         private boolean advance() {
            while (colItr == null || !colItr.hasNext()) {
               if (colItr == null && !rowItr.hasNext()) {
                  return false;
               } else if (colItr == null) {
                  row = rowItr.next();
                  colItr = IntStream.range(0, numberOfColumns()).iterator();
                  return true;
               } else if (!colItr.hasNext()) {
                  colItr = null;
               }
            }
            return colItr != null;
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
            int col = colItr.next();
            return new Matrix.Entry(row, col, get(row, col));
         }
      };
   }

   /**
    * Converts the matrix to a dense matrix.
    *
    * @return the dense matrix
    */
   DenseMatrix toDense();


   /**
    * For each sparse.
    *
    * @param consumer the consumer
    */
   default void forEachSparse(@NonNull Consumer<Matrix.Entry> consumer) {
      nonZeroIterator().forEachRemaining(consumer);
   }

   /**
    * For each ordered sparse.
    *
    * @param consumer the consumer
    */
   default void forEachOrderedSparse(@NonNull Consumer<Matrix.Entry> consumer) {
      orderedNonZeroIterator().forEachRemaining(consumer);
   }

   /**
    * Map matrix.
    *
    * @param operator the operator
    * @return the matrix
    */
   default Matrix map(@NonNull DoubleUnaryOperator operator) {
      return copy().mapSelf(operator);
   }

   /**
    * Map self matrix.
    *
    * @param operator the operator
    * @return the matrix
    */
   default Matrix mapSelf(@NonNull DoubleUnaryOperator operator) {
      forEach(entry -> set(entry.row, entry.column, operator.applyAsDouble(entry.value)));
      return this;
   }

   /**
    * Non zero iterator iterator.
    *
    * @return the iterator
    */
   default Iterator<Matrix.Entry> nonZeroIterator() {
      return orderedNonZeroIterator();
   }

   /**
    * Ordered non zero iterator iterator.
    *
    * @return the iterator
    */
   default Iterator<Matrix.Entry> orderedNonZeroIterator() {
      return new Iterator<Entry>() {
         private PrimitiveIterator.OfInt rowItr = IntStream.range(0, numberOfRows()).iterator();
         private int row;
         private Integer col;
         private PrimitiveIterator.OfInt colItr = null;

         private boolean advance() {
            while (col == null || get(row, col) == 0) {

               if (colItr == null && !rowItr.hasNext()) {
                  return false;
               }

               if (colItr == null) {
                  row = rowItr.next();
                  colItr = IntStream.range(0, numberOfColumns()).iterator();
                  col = colItr.next();
               } else if (!colItr.hasNext()) {
                  colItr = null;
               } else {
                  col = colItr.next();
               }

            }

            return col != null && get(row, col) != 0;
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
            int column = col;
            col = null;
            return new Matrix.Entry(row, column, get(row, column));
         }
      };
   }


   /**
    * Gets column.
    *
    * @param column the column
    * @return the column
    */
   Vector column(int column);

   /**
    * Gets row.
    *
    * @param row the row
    * @return the row
    */
   Vector row(int row);

   /**
    * Get double.
    *
    * @param row    the row
    * @param column the column
    * @return the double
    */
   double get(int row, int column);


   /**
    * Row iterator iterator.
    *
    * @return the iterator
    */
   Iterator<Vector> rowIterator();

   /**
    * Column iterator iterator.
    *
    * @return the iterator
    */
   Iterator<Vector> columnIterator();

   /**
    * For each row.
    *
    * @param consumer the consumer
    */
   default void forEachRow(@NonNull Consumer<Vector> consumer) {
      rowIterator().forEachRemaining(consumer);
   }

   /**
    * For each column.
    *
    * @param consumer the consumer
    */
   default void forEachColumn(@NonNull Consumer<Vector> consumer) {
      columnIterator().forEachRemaining(consumer);
   }

   /**
    * Dot vector.
    *
    * @param vector the vector
    * @return the vector
    */
   default Vector dot(@NonNull Vector vector) {
      Preconditions.checkArgument(vector.dimension() == numberOfColumns(), "Dimension mismatch");
      SparseVector output = new SparseVector(numberOfRows());
      for (int r = 0; r < numberOfRows(); r++) {
         output.set(r, row(r).dot(vector));
      }
      return output;
   }

   /**
    * Set void.
    *
    * @param row    the row
    * @param column the column
    * @param value  the value
    */
   void set(int row, int column, double value);

   /**
    * Sets column vector.
    *
    * @param column the column
    * @param vector the vector
    */
   void setColumn(int column, Vector vector);

   /**
    * Sets row vector.
    *
    * @param row    the row
    * @param vector the vector
    */
   void setRow(int row, Vector vector);

   /**
    * To array.
    *
    * @return the double [ ] [ ]
    */
   double[][] toArray();

   /**
    * Row k.
    *
    * @return the int
    */
   int numberOfRows();

   /**
    * Column k.
    *
    * @return the int
    */
   int numberOfColumns();

   /**
    * Add matrix.
    *
    * @param other the other
    * @return the matrix
    */
   default Matrix add(@NonNull Matrix other) {
      return copy().addSelf(other);
   }

   /**
    * Add self matrix.
    *
    * @param other the other
    * @return the matrix
    */
   Matrix addSelf(Matrix other);

   /**
    * Subtract matrix.
    *
    * @param other the other
    * @return the matrix
    */
   default Matrix subtract(@NonNull Matrix other) {
      return copy().subtractSelf(other);
   }

   /**
    * Subtract self matrix.
    *
    * @param other the other
    * @return the matrix
    */
   Matrix subtractSelf(Matrix other);

   /**
    * Scale matrix.
    *
    * @param other the other
    * @return the matrix
    */
   default Matrix scale(@NonNull Matrix other) {
      return copy().scaleSelf(other);
   }

   /**
    * Scale self matrix.
    *
    * @param other the other
    * @return the matrix
    */
   Matrix scaleSelf(Matrix other);


   /**
    * Scale self.
    *
    * @param value the value
    * @return the matrix
    */
   Matrix scaleSelf(double value);

   /**
    * Scale matrix.
    *
    * @param value the value
    * @return the matrix
    */
   default Matrix scale(double value) {
      return copy().scaleSelf(value);
   }

   /**
    * Increment self.
    *
    * @param value the value
    * @return the matrix
    */
   Matrix incrementSelf(double value);

   /**
    * Increment matrix.
    *
    * @param value the value
    * @return the matrix
    */
   default Matrix increment(double value) {
      return copy().incrementSelf(value);
   }

   /**
    * Multiply matrix.
    *
    * @param m the m
    * @return the matrix
    */
   Matrix multiply(Matrix m);

   /**
    * Transpose matrix.
    *
    * @return the matrix
    */
   Matrix transpose();

   /**
    * Is sparse.
    *
    * @return the boolean
    */
   boolean isSparse();

   /**
    * Increment matrix.
    *
    * @param row the row
    * @param col the col
    * @return the matrix
    */
   default Matrix increment(int row, int col) {
      return increment(row, col, 1);
   }

   /**
    * Increment matrix.
    *
    * @param row    the row
    * @param col    the col
    * @param amount the amount
    * @return the matrix
    */
   Matrix increment(int row, int col, double amount);

   /**
    * Decrement matrix.
    *
    * @param row    the row
    * @param col    the col
    * @param amount the amount
    * @return the matrix
    */
   default Matrix decrement(int row, int col, double amount) {
      return increment(row, col, -amount);
   }

   /**
    * Decrement matrix.
    *
    * @param row the row
    * @param col the col
    * @return the matrix
    */
   default Matrix decrement(int row, int col) {
      return increment(row, col, -1);
   }

   /**
    * Scale matrix.
    *
    * @param r      the r
    * @param c      the c
    * @param amount the amount
    * @return the matrix
    */
   default Matrix scale(int r, int c, double amount) {
      set(r, c, get(r, c) * amount);
      return this;
   }


   /**
    * Diag vector vector.
    *
    * @return the vector
    */
   default Vector diagVector() {
      Vector diag = new DenseVector(Math.max(numberOfColumns(), numberOfRows()));
      for (int r = 0; r < numberOfRows() && r < numberOfColumns(); r++) {
         diag.set(r, get(r, r));
      }
      return diag;
   }

   /**
    * Diag matrix.
    *
    * @return the matrix
    */
   Matrix diag();

   /**
    * Sum double.
    *
    * @return the double
    */
   default double sum() {
      return Streams.asStream(this).mapToDouble(Entry::getValue).sum();
   }

   /**
    * The type Entry.
    */
   @Value
   class Entry {
      /**
       * The Row.
       */
      public int row;
      /**
       * The Column.
       */
      public int column;
      /**
       * The Value.
       */
      public double value;

   }

}//END OF Matrix
