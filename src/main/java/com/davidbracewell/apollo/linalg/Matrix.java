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
import lombok.NonNull;
import lombok.Value;

import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.PrimitiveIterator;
import java.util.function.Consumer;
import java.util.stream.IntStream;

/**
 * The interface Matrix.
 *
 * @author David B. Bracewell
 */
public interface Matrix extends Copyable<Matrix>, Iterable<Matrix.Entry> {


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
   * Dot vector.
   *
   * @param v the v
   * @return the vector
   */
  Vector dot(Vector v);

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
   * Row dimension.
   *
   * @return the int
   */
  int numberOfRows();

  /**
   * Column dimension.
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
  default Matrix multiply(@NonNull Matrix m) {
    return copy().multiplySelf(m);
  }

  /**
   * Multiply self matrix.
   *
   * @param m the m
   * @return the matrix
   */
  Matrix multiplySelf(Matrix m);

  /**
   * Multiply matrix.
   *
   * @param v the v
   * @return the matrix
   */
  default Matrix multiply(@NonNull Vector v) {
    return copy().multiplySelf(v);
  }

  /**
   * Multiply self matrix.
   *
   * @param v the v
   * @return the matrix
   */
  Matrix multiplySelf(Vector v);

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
