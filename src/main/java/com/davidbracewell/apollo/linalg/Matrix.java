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
import com.davidbracewell.conversion.Cast;
import com.davidbracewell.function.SerializableFunction;
import com.davidbracewell.guava.common.base.Preconditions;
import com.davidbracewell.tuple.Tuple2;
import lombok.NonNull;
import lombok.Value;

import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.PrimitiveIterator;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;
import java.util.function.DoubleBinaryOperator;
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
    * Transpose matrix.
    *
    * @return the matrix
    */
   default Matrix T() {
      final Matrix transposed = getFactory().create(numberOfColumns(), numberOfRows());
      forEachSparse(e -> transposed.set(e.column, e.row, e.value));
      return transposed;
   }

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
    * Add column matrix.
    *
    * @param vector the vector
    * @return the matrix
    */
   default Matrix addColumn(@NonNull Vector vector) {
      return copy().addColumnSelf(vector);
   }

   /**
    * Add column self matrix.
    *
    * @param vector the vector
    * @return the matrix
    */
   default Matrix addColumnSelf(@NonNull Vector vector) {
      forEachColumn(c -> c.addSelf(vector));
      return this;
   }

   /**
    * Add row matrix.
    *
    * @param vector the vector
    * @return the matrix
    */
   default Matrix addRow(@NonNull Vector vector) {
      return copy().addRowSelf(vector);
   }

   /**
    * Add row self matrix.
    *
    * @param vector the vector
    * @return the matrix
    */
   default Matrix addRowSelf(@NonNull Vector vector) {
      forEachRow(r -> r.addSelf(vector));
      return this;
   }

   /**
    * Add self matrix.
    *
    * @param other the other
    * @return the matrix
    */
   default Matrix addSelf(@NonNull Matrix other) {
      Preconditions.checkArgument(
         other.numberOfColumns() == numberOfColumns() && other.numberOfRows() == numberOfRows(),
         "Dimension Mismatch");
      other.forEachSparse(e -> increment(e.row, e.column, e.value));
      return this;
   }

   /**
    * Gets column.
    *
    * @param column the column
    * @return the column
    */
   default Vector column(int column) {
      Preconditions.checkPositionIndex(column, numberOfColumns(), "Column out of index");
      return new ColumnVector(this, column);
   }

   /**
    * Column iterator iterator.
    *
    * @return the iterator
    */
   default Iterator<Vector> columnIterator() {
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
    * Diag matrix.
    *
    * @return the matrix
    */
   default Matrix diag() {
      Matrix m = getFactory().create(numberOfRows(), numberOfColumns());
      for (int r = 0; r < numberOfRows() && r < numberOfColumns(); r++) {
         m.set(r, r, get(r, r));
      }
      return m;
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
    * For each column.
    *
    * @param consumer the consumer
    */
   default void forEachColumn(@NonNull Consumer<Vector> consumer) {
      columnIterator().forEachRemaining(consumer);
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
    * For each row.
    *
    * @param consumer the consumer
    */
   default void forEachRow(@NonNull Consumer<Vector> consumer) {
      rowIterator().forEachRemaining(consumer);
   }

   /**
    * For each sparse.
    *
    * @param consumer the consumer
    */
   default void forEachSparse(@NonNull Consumer<Matrix.Entry> consumer) {
      nonZeroIterator().forEachRemaining(consumer);
   }

   /**
    * Get double.
    *
    * @param row    the row
    * @param column the column
    * @return the double
    */
   double get(int row, int column);

   /**
    * Gets factory.
    *
    * @return the factory
    */
   MatrixFactory getFactory();

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
    * Increment self.
    *
    * @param value the value
    * @return the matrix
    */
   default Matrix incrementSelf(double value) {
      forEachRow(row -> row.mapAddSelf(value));
      return this;
   }

   /**
    * Is sparse.
    *
    * @return the boolean
    */
   boolean isSparse();

   @Override
   default Iterator<Entry> iterator() {
      return new Iterator<Entry>() {
         private PrimitiveIterator.OfInt rowItr = IntStream
                                                     .range(0, numberOfRows())
                                                     .iterator();
         private int row;
         private PrimitiveIterator.OfInt colItr = null;

         private boolean advance() {
            while (colItr == null || !colItr.hasNext()) {
               if (colItr == null && !rowItr.hasNext()) {
                  return false;
               } else if (colItr == null) {
                  row = rowItr.next();
                  colItr = IntStream
                              .range(0, numberOfColumns())
                              .iterator();
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
    * Map matrix.
    *
    * @param operator the operator
    * @return the matrix
    */
   default Matrix map(@NonNull DoubleUnaryOperator operator) {
      return copy().mapSelf(operator);
   }

   /**
    * Map column matrix.
    *
    * @param vector   the vector
    * @param operator the operator
    * @return the matrix
    */
   default Matrix mapColumn(@NonNull Vector vector, @NonNull DoubleBinaryOperator operator) {
      return copy().mapColumnSelf(vector, operator);
   }

   /**
    * Map column self matrix.
    *
    * @param vector   the vector
    * @param operator the operator
    * @return the matrix
    */
   default Matrix mapColumnSelf(@NonNull Vector vector, @NonNull DoubleBinaryOperator operator) {
      forEachColumn(r -> r.mapSelf(vector, operator));
      return this;
   }

   /**
    * Map row matrix.
    *
    * @param vector   the vector
    * @param operator the operator
    * @return the matrix
    */
   default Matrix mapRow(@NonNull Vector vector, @NonNull DoubleBinaryOperator operator) {
      return copy().mapRowSelf(vector, operator);
   }

   /**
    * Map row matrix.
    *
    * @param function the function
    * @return the matrix
    */
   default Matrix mapRow(@NonNull SerializableFunction<Vector, Vector> function) {
      Matrix mPrime = copy();
      for (int i = 0; i < mPrime.numberOfRows(); i++) {
         mPrime.setRow(i, function.apply(row(i)));
      }
      return mPrime;
   }

   /**
    * Map row self matrix.
    *
    * @param vector   the vector
    * @param operator the operator
    * @return the matrix
    */
   default Matrix mapRowSelf(@NonNull Vector vector, @NonNull DoubleBinaryOperator operator) {
      forEachRow(r -> r.mapSelf(vector, operator));
      return this;
   }

   /**
    * Map row self matrix.
    *
    * @param function the function
    * @return the matrix
    */
   default Matrix mapRowSelf(@NonNull SerializableFunction<Vector, Vector> function) {
      for (int i = 0; i < numberOfRows(); i++) {
         setRow(i, function.apply(row(i)));
      }
      return this;
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
    * Multiply matrix.
    *
    * @param m the m
    * @return the matrix
    */
   default Matrix multiply(@NonNull Matrix m) {
      Preconditions.checkArgument(numberOfColumns() == m.numberOfRows(), "Dimension Mismatch");
      if (m.isSparse() && isSparse()) {
         return new SparseMatrix(Cast.<SparseMatrix>as(this).asRealMatrix()
                                                            .multiply(Cast.<SparseMatrix>as(m).asRealMatrix()));
      }
      Matrix mprime = getFactory().create(numberOfRows(), m.numberOfColumns());
      IntStream
         .range(0, numberOfRows())
         .parallel()
         .forEach(r -> {
            for (int c = 0; c < m.numberOfColumns(); c++) {
               for (int k = 0; k < numberOfColumns(); k++) {
                  mprime.increment(r, c, get(r, k) * m.get(k, c));
               }
            }
         });
      return mprime;
   }

   /**
    * Multiply vector matrix.
    *
    * @param v the v
    * @return the matrix
    */
   default Matrix multiplyVector(@NonNull Vector v) {
      return copy().multiplyVectorSelf(v);
   }

   /**
    * Multiply vector self matrix.
    *
    * @param v the v
    * @return the matrix
    */
   default Matrix multiplyVectorSelf(@NonNull Vector v) {
      rowIterator().forEachRemaining(r -> r.multiply(v));
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
    * Column dimension.
    *
    * @return the int
    */
   int numberOfColumns();

   /**
    * Row dimension.
    *
    * @return the int
    */
   int numberOfRows();

   /**
    * Ordered non zero iterator iterator.
    *
    * @return the iterator
    */
   default Iterator<Matrix.Entry> orderedNonZeroIterator() {
      return new Iterator<Entry>() {
         private PrimitiveIterator.OfInt rowItr = IntStream
                                                     .range(0, numberOfRows())
                                                     .iterator();
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
                  colItr = IntStream
                              .range(0, numberOfColumns())
                              .iterator();
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
    * Gets row.
    *
    * @param row the row
    * @return the row
    */
   default Vector row(int row) {
      Preconditions.checkPositionIndex(row, numberOfRows(), "Row out of index");
      return new RowVector(this, row);
   }

   /**
    * Row iterator iterator.
    *
    * @return the iterator
    */
   default Iterator<Vector> rowIterator() {
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
    * Scale matrix.
    *
    * @param value the value
    * @return the matrix
    */
   default Matrix scale(double value) {
      return copy().scaleSelf(value);
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
    * Scale self matrix.
    *
    * @param other the other
    * @return the matrix
    */
   default Matrix scaleSelf(Matrix other) {
      Preconditions.checkArgument(
         other.numberOfColumns() == numberOfColumns() && other.numberOfRows() == numberOfRows(),
         "Dimension Mismatch");
      other.forEachSparse(e -> scale(e.row, e.column, e.value));
      return this;
   }

   /**
    * Scale self.
    *
    * @param value the value
    * @return the matrix
    */
   default Matrix scaleSelf(double value) {
      forEachSparse(e -> set(e.row, e.column, e.value * value));
      return this;
   }

   /**
    * Set void.
    *
    * @param row    the row
    * @param column the column
    * @param value  the value
    * @return the matrix
    */
   Matrix set(int row, int column, double value);

   /**
    * Sets column vector.
    *
    * @param column the column
    * @param vector the vector
    * @return the column
    */
   default Matrix setColumn(int column, @NonNull Vector vector) {
      Preconditions.checkElementIndex(column, numberOfColumns());
      Preconditions.checkArgument(vector.dimension() == numberOfRows(), "Dimension Mismatch");
      vector.forEach(e -> set(e.index, column, e.value));
      return this;
   }

   /**
    * Sets row vector.
    *
    * @param row    the row
    * @param vector the vector
    * @return the row
    */
   default Matrix setRow(int row, Vector vector) {
      Preconditions.checkElementIndex(row, numberOfRows());
      Preconditions.checkArgument(vector.dimension() == numberOfColumns(), "Dimension Mismatch");
      vector.forEach(e -> set(row, e.index, e.value));
      return this;
   }

   /**
    * The shape of the matrix as <code>number of rows</code>, <code>number of columns</code> tuple
    *
    * @return the shape
    */
   default Tuple2<Integer, Integer> shape() {
      return $(numberOfRows(), numberOfColumns());
   }

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
    * Subtract column matrix.
    *
    * @param vector the vector
    * @return the matrix
    */
   default Matrix subtractColumn(@NonNull Vector vector) {
      return copy().subtractColumnSelf(vector);
   }

   /**
    * Subtract column self matrix.
    *
    * @param vector the vector
    * @return the matrix
    */
   default Matrix subtractColumnSelf(@NonNull Vector vector) {
      forEachColumn(c -> c.subtractSelf(vector));
      return this;
   }

   /**
    * Subtract row matrix.
    *
    * @param vector the vector
    * @return the matrix
    */
   default Matrix subtractRow(@NonNull Vector vector) {
      return copy().subtractRowSelf(vector);
   }

   /**
    * Subtract row self matrix.
    *
    * @param vector the vector
    * @return the matrix
    */
   default Matrix subtractRowSelf(@NonNull Vector vector) {
      forEachRow(r -> r.subtractSelf(vector));
      return this;
   }

   /**
    * Subtract self matrix.
    *
    * @param other the other
    * @return the matrix
    */
   default Matrix subtractSelf(@NonNull Matrix other) {
      Preconditions.checkArgument(
         other.numberOfColumns() == numberOfColumns() && other.numberOfRows() == numberOfRows(),
         "Dimension Mismatch");
      other.forEachSparse(e -> decrement(e.row, e.column, e.value));
      return this;
   }

   /**
    * Sum double.
    *
    * @return the double
    */
   default double sum() {
      return Streams
                .asStream(this)
                .mapToDouble(Entry::getValue)
                .sum();
   }

   /**
    * To array.
    *
    * @return the double [ ] [ ]
    */
   default double[][] toArray() {
      double[][] array = new double[numberOfRows()][numberOfColumns()];
      forEachSparse(e -> array[e.row][e.column] = e.value);
      return array;
   }

   /**
    * Converts the matrix to a dense matrix.
    *
    * @return the dense matrix
    */
   default DenseMatrix toDense() {
      return new DenseMatrix(this);
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
