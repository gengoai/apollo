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
import com.davidbracewell.stream.Streams;
import com.google.common.base.Preconditions;
import lombok.NonNull;
import org.apache.mahout.math.map.OpenIntObjectHashMap;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.PrimitiveIterator;
import java.util.stream.IntStream;

/**
 * @author David B. Bracewell
 */
public class SparseMatrix implements Matrix, Serializable {
  private static final long serialVersionUID = -3802597548916836308L;
  final private int numberOfRows;
  final private int colDimension;
  private volatile OpenIntObjectHashMap<Vector> matrix;

  public SparseMatrix(int numRows, int numColumns) {
    this.colDimension = numColumns;
    this.numberOfRows = numRows;
    this.matrix = new OpenIntObjectHashMap<>();
  }

  public SparseMatrix(@NonNull Matrix matrix) {
    this(matrix.numberOfRows(), matrix.numberOfColumns());
    matrix.forEachSparse(e -> set(e.row, e.column, e.value));
  }

  public SparseMatrix(Vector[] vectors) {
    this(Arrays.asList(Preconditions.checkNotNull(vectors)));
  }

  public SparseMatrix(List<Vector> vectors) {
    Preconditions.checkNotNull(vectors);
    if (vectors.size() == 0) {
      this.colDimension = 0;
      this.numberOfRows = 0;
    } else {
      this.colDimension = vectors.get(0).dimension();
      this.numberOfRows = vectors.size();
    }
    this.matrix = new OpenIntObjectHashMap<>();
    for (int i = 0; i < vectors.size(); i++) {
      this.matrix.put(i, vectors.get(i));
    }
  }

  public static Matrix zeroes(int numberOfRows, int numberOfColumns) {
    return new SparseMatrix(numberOfRows, numberOfColumns);
  }

  public static Matrix ones(int numberOfRows, int numberOfColumns) {
    return new SparseMatrix(numberOfRows, numberOfColumns).incrementSelf(1);
  }

  public static Matrix unit(int size) {
    Matrix m = new SparseMatrix(size, size);
    for (int r = 0; r < size; r++) {
      m.set(r, r, 1d);
    }
    return m;
  }

  public static Matrix random(int numberOfRows, int numberOfColumns) {
    Matrix m = new SparseMatrix(numberOfRows, numberOfColumns);
    Streams.range(0, numberOfRows)
      .parallel()
      .forEach(r -> {
        for (int c = 0; c < numberOfColumns; c++) {
          m.set(r, c, Math.random());
        }
      });
    return m;
  }

  public static void main(String[] args) {
    Matrix m1 = SparseMatrix.random(1000, 1000);
    Matrix m2 = SparseMatrix.random(1000, 1000);
    System.out.println(m1.multiply(m2));
  }

  @Override
  public DenseMatrix toDense() {
    return new DenseMatrix(this);
  }

  @Override
  public Iterator<Entry> nonZeroIterator() {
    return new Iterator<Entry>() {
      private PrimitiveIterator.OfInt rowItr = IntStream.of(matrix.keys().toArray(new int[matrix.size()])).iterator();
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
      private PrimitiveIterator.OfInt rowItr = IntStream.of(matrix.keys().toArray(new int[matrix.size()])).sorted().iterator();
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
  public Vector column(int column) {
    Preconditions.checkElementIndex(column, numberOfColumns());
    SparseVector col = new SparseVector(numberOfRows());
    for (int row : matrix.keys().elements()) {
      col.set(row, get(row, column));
    }
    return col;
  }

  @Override
  public Vector row(int row) {
    Preconditions.checkElementIndex(row, numberOfRows());
    if (!matrix.containsKey(row)) {
      synchronized (this) {
        if (!matrix.containsKey(row)) {
          matrix.put(row, new SparseVector(numberOfColumns()));
        }
      }
    }
    return matrix.get(row);
  }

  @Override
  public double get(int row, int column) {
    Preconditions.checkElementIndex(row, numberOfRows());
    Preconditions.checkElementIndex(column, numberOfColumns());
    return row(row).get(column);
  }

  @Override
  public void set(int row, int column, double value) {
    Preconditions.checkElementIndex(row, numberOfRows());
    Preconditions.checkElementIndex(column, numberOfColumns());
    row(row).set(column, value);
  }

  @Override
  public void setColumn(int column, Vector vector) {
    Preconditions.checkNotNull(vector);
    Preconditions.checkArgument(column >= 0 && column < numberOfColumns());
    Preconditions.checkArgument(vector.dimension() == numberOfRows());
    for (Vector.Entry entry : Collect.asIterable(vector.nonZeroIterator())) {
      set(entry.index, column, entry.value);
    }
  }

  @Override
  public void setRow(int row, @NonNull Vector vector) {
    Preconditions.checkElementIndex(row, numberOfRows);
    Preconditions.checkArgument(vector.dimension() == numberOfColumns());
    synchronized (matrix) {
      matrix.put(row, vector);
    }
  }

  @Override
  public double[][] toArray() {
    double[][] m = new double[numberOfRows()][numberOfColumns()];
    for (int row : matrix.keys().elements()) {
      for (Vector.Entry entry : Collect.asIterable(matrix.get(row).nonZeroIterator())) {
        m[row][entry.index] = entry.value;
      }
    }
    return m;
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
  public Matrix addSelf(@NonNull Matrix other) {
    Preconditions.checkArgument(numberOfRows() == other.numberOfRows() && numberOfColumns() == other.numberOfColumns(), "Dimension mismatch");
    for (int r = 0; r < numberOfRows(); r++) {
      row(r).addSelf(other.row(r));
    }
    return this;
  }

  @Override
  public Matrix subtractSelf(@NonNull Matrix other) {
    Preconditions.checkArgument(numberOfRows() == other.numberOfRows() && numberOfColumns() == other.numberOfColumns(), "Dimension mismatch");
    other.forEachSparse(e -> decrement(e.row, e.column, e.value));
    return this;
  }

  @Override
  public Matrix scaleSelf(@NonNull Matrix other) {
    Preconditions.checkArgument(numberOfRows() == other.numberOfRows() && numberOfColumns() == other.numberOfColumns(), "Dimension mismatch");
    other.forEachSparse(e -> scale(e.row, e.column, e.value));
    return this;
  }

  @Override
  public Matrix scaleSelf(@NonNull Vector other) {
    Preconditions.checkArgument(numberOfColumns() == other.dimension(), "Dimension mismatch");
    matrix.keys().forEach(r -> {
      row(r).multiplySelf(other);
      return true;
    });
    return this;
  }

  @Override
  public Matrix scaleSelf(double value) {
    for (int row : matrix.keys().elements()) {
      for (Vector.Entry entry : Collect.asIterable(row(row).nonZeroIterator())) {
        row(row).set(entry.index, entry.value * value);
      }
    }
    return this;
  }

  @Override
  public Matrix incrementSelf(double value) {
    for (int r = 0; r < numberOfRows(); r++) {
      row(r).mapAddSelf(value);
    }
    return this;
  }

  @Override
  public Matrix multiply(@NonNull Matrix m) {
    Preconditions.checkArgument(numberOfColumns() == m.numberOfRows(), "Dimension Mismatch");
    SparseMatrix mprime = new SparseMatrix(numberOfRows(), m.numberOfColumns());
    IntStream.of(matrix.keys().toArray(new int[matrix.size()])).parallel()
      .forEach(r -> {
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
    Matrix T = new SparseMatrix(numberOfColumns(), numberOfRows());
    forEachSparse(e -> T.set(e.column, e.row, e.value));
    return T;
  }

  @Override
  public boolean isSparse() {
    return true;
  }

  @Override
  public Matrix increment(int row, int col, double amount) {
    Preconditions.checkElementIndex(row, numberOfRows());
    Preconditions.checkElementIndex(col, numberOfColumns());
    row(row).increment(col, amount);
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
