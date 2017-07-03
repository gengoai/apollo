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
import com.davidbracewell.function.SerializableFunction;
import com.davidbracewell.tuple.Tuple2;
import lombok.NonNull;
import lombok.Value;

import java.util.Iterator;
import java.util.function.Consumer;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;

/**
 * <p>Interface for Matrices</p>
 *
 * @author David B. Bracewell
 */
public interface Matrix extends Copyable<Matrix>, Iterable<Matrix.Entry> {


   static Vector fold(@NonNull Matrix... matrices) {
      int dimension = 0;
      for (Matrix matrix : matrices) {
         dimension += (matrix.numberOfRows() * matrix.numberOfColumns());
      }
      Vector vector = Vector.dZeros(dimension);
      int offset = 0;
      for (Matrix matrix : matrices) {
         for (int c = 0; c < matrix.numberOfColumns(); c++) {
            for (int r = 0; r < matrix.numberOfRows(); r++) {
               vector.set(offset++, matrix.get(r, c));
            }
         }
      }
      return vector;
   }

   static Matrix[] unfold(@NonNull Vector vector, @NonNull int[][] shapes) {
      Matrix[] toReturn = new Matrix[shapes.length];
      for (int i = 0; i < shapes.length; i++) {
         toReturn[i] = new DenseMatrix(shapes[i][0], shapes[i][1]);
      }

      int index = 0;
      for (int i = 0; i < shapes.length; i++) {
         int numberOfRows = shapes[i][0];
         int numberOfColumns = shapes[i][1];
         for (int c = 0; c < numberOfColumns; c++) {
            for (int r = 0; r < numberOfRows; r++) {
               toReturn[i].set(r, c, vector.get(index++));
            }
         }
      }

      return toReturn;
   }

   /**
    * Transpose matrix.
    *
    * @return the matrix
    */
   Matrix T();

   /**
    * Add matrix.
    *
    * @param other the other
    * @return the matrix
    */
   Matrix add(@NonNull Matrix other);

   /**
    * Add column matrix.
    *
    * @param vector the vector
    * @return the matrix
    */
   Matrix addColumn(@NonNull Vector vector);

   /**
    * Add column self matrix.
    *
    * @param vector the vector
    * @return the matrix
    */
   Matrix addColumnSelf(@NonNull Vector vector);

   /**
    * Add row matrix.
    *
    * @param vector the vector
    * @return the matrix
    */
   Matrix addRow(@NonNull Vector vector);

   /**
    * Add row self matrix.
    *
    * @param vector the vector
    * @return the matrix
    */
   Matrix addRowSelf(@NonNull Vector vector);

   /**
    * Add self matrix.
    *
    * @param other the other
    * @return the matrix
    */
   Matrix addSelf(@NonNull Matrix other);

   /**
    * Gets column.
    *
    * @param column the column
    * @return the column
    */
   Vector column(int column);

   /**
    * Column iterator iterator.
    *
    * @return the iterator
    */
   Iterator<Vector> columnIterator();

   /**
    * Decrement matrix.
    *
    * @param row    the row
    * @param col    the col
    * @param amount the amount
    * @return the matrix
    */
   Matrix decrement(int row, int col, double amount);

   /**
    * Decrement matrix.
    *
    * @param row the row
    * @param col the col
    * @return the matrix
    */
   Matrix decrement(int row, int col);

   /**
    * Diag matrix.
    *
    * @return the matrix
    */
   Matrix diag();

   /**
    * Diag vector vector.
    *
    * @return the vector
    */
   Vector diagVector();

   /**
    * Dot vector.
    *
    * @param vector the vector
    * @return the vector
    */
   Matrix dot(Vector vector);

   Matrix dot(Matrix matrix);

   Matrix ebeMultiply(Matrix matrix);

   Matrix ebeMultiplySelf(Matrix matrix);

   /**
    * For each column.
    *
    * @param consumer the consumer
    */
   void forEachColumn(@NonNull Consumer<Vector> consumer);

   /**
    * For each ordered sparse.
    *
    * @param consumer the consumer
    */
   void forEachOrderedSparse(@NonNull Consumer<Matrix.Entry> consumer);

   /**
    * For each row.
    *
    * @param consumer the consumer
    */
   void forEachRow(@NonNull Consumer<Vector> consumer);

   /**
    * For each sparse.
    *
    * @param consumer the consumer
    */
   void forEachSparse(@NonNull Consumer<Matrix.Entry> consumer);

   /**
    * Get double.
    *
    * @param row    the row
    * @param column the column
    * @return the double
    */
   double get(int row, int column);

   /**
    * Increment matrix.
    *
    * @param value the value
    * @return the matrix
    */
   Matrix increment(double value);

   /**
    * Increment matrix.
    *
    * @param row the row
    * @param col the col
    * @return the matrix
    */
   Matrix increment(int row, int col);

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
   Matrix incrementSelf(double value);

   boolean isDense();

   /**
    * Is sparse.
    *
    * @return the boolean
    */
   boolean isSparse();

   @Override
   Iterator<Entry> iterator();

   /**
    * Map matrix.
    *
    * @param operator the operator
    * @return the matrix
    */
   Matrix map(@NonNull DoubleUnaryOperator operator);

   /**
    * Map column matrix.
    *
    * @param vector   the vector
    * @param operator the operator
    * @return the matrix
    */
   Matrix mapColumn(@NonNull Vector vector, @NonNull DoubleBinaryOperator operator);

   /**
    * Map column self matrix.
    *
    * @param vector   the vector
    * @param operator the operator
    * @return the matrix
    */
   Matrix mapColumnSelf(@NonNull Vector vector, @NonNull DoubleBinaryOperator operator);

   /**
    * Map row matrix.
    *
    * @param vector   the vector
    * @param operator the operator
    * @return the matrix
    */
   Matrix mapRow(@NonNull Vector vector, @NonNull DoubleBinaryOperator operator);

   /**
    * Map row matrix.
    *
    * @param function the function
    * @return the matrix
    */
   Matrix mapRow(@NonNull SerializableFunction<Vector, Vector> function);

   /**
    * Map row self matrix.
    *
    * @param vector   the vector
    * @param operator the operator
    * @return the matrix
    */
   Matrix mapRowSelf(@NonNull Vector vector, @NonNull DoubleBinaryOperator operator);

   /**
    * Map row self matrix.
    *
    * @param function the function
    * @return the matrix
    */
   Matrix mapRowSelf(@NonNull SerializableFunction<Vector, Vector> function);

   /**
    * Map self matrix.
    *
    * @param operator the operator
    * @return the matrix
    */
   Matrix mapSelf(@NonNull DoubleUnaryOperator operator);

   /**
    * Multiply matrix.
    *
    * @param m the m
    * @return the matrix
    */
   Matrix multiply(@NonNull Matrix m);

   /**
    * Multiply vector matrix.
    *
    * @param v the v
    * @return the matrix
    */
   Matrix multiplyVector(@NonNull Vector v);

   /**
    * Multiply vector self matrix.
    *
    * @param v the v
    * @return the matrix
    */
   Matrix multiplyVectorSelf(@NonNull Vector v);

   /**
    * Non zero iterator iterator.
    *
    * @return the iterator
    */
   Iterator<Matrix.Entry> nonZeroIterator();

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
   Iterator<Matrix.Entry> orderedNonZeroIterator();

   /**
    * Gets row.
    *
    * @param row the row
    * @return the row
    */
   Vector row(int row);

   /**
    * Row iterator iterator.
    *
    * @return the iterator
    */
   Iterator<Vector> rowIterator();

   /**
    * Scale matrix.
    *
    * @param other the other
    * @return the matrix
    */
   Matrix scale(@NonNull Matrix other);

   /**
    * Scale matrix.
    *
    * @param value the value
    * @return the matrix
    */
   Matrix scale(double value);

   /**
    * Scale matrix.
    *
    * @param r      the r
    * @param c      the c
    * @param amount the amount
    * @return the matrix
    */
   Matrix scale(int r, int c, double amount);

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
   Matrix setColumn(int column, @NonNull Vector vector);

   /**
    * Sets row vector.
    *
    * @param row    the row
    * @param vector the vector
    * @return the row
    */
   Matrix setRow(int row, Vector vector);

   /**
    * The shape of the matrix as <code>number of rows</code>, <code>number of columns</code> tuple
    *
    * @return the shape
    */
   Tuple2<Integer, Integer> shape();

   /**
    * Subtract matrix.
    *
    * @param other the other
    * @return the matrix
    */
   Matrix subtract(@NonNull Matrix other);

   /**
    * Subtract column matrix.
    *
    * @param vector the vector
    * @return the matrix
    */
   Matrix subtractColumn(@NonNull Vector vector);

   /**
    * Subtract column self matrix.
    *
    * @param vector the vector
    * @return the matrix
    */
   Matrix subtractColumnSelf(@NonNull Vector vector);

   /**
    * Subtract row matrix.
    *
    * @param vector the vector
    * @return the matrix
    */
   Matrix subtractRow(@NonNull Vector vector);

   /**
    * Subtract row self matrix.
    *
    * @param vector the vector
    * @return the matrix
    */
   Matrix subtractRowSelf(@NonNull Vector vector);

   /**
    * Subtract self matrix.
    *
    * @param other the other
    * @return the matrix
    */
   Matrix subtractSelf(@NonNull Matrix other);

   /**
    * Sum double.
    *
    * @return the double
    */
   double sum();

   /**
    * To array.
    *
    * @return the double [ ] [ ]
    */
   double[][] toArray();

   /**
    * Converts the matrix to a dense matrix.
    *
    * @return the dense matrix
    */
   DenseMatrix toDense();

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
