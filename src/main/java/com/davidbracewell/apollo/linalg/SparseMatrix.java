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
  private final OpenIntObjectHashMap<Vector> matrix;
  final private int numberOfRows;
  final private int colDimension;
  private double nonZero = 0d;

  public SparseMatrix(int numRows, int numColumns) {
    this(numRows, numColumns, 0d);
  }

  public SparseMatrix(int numRows, int numColumns, double nonZero) {
    this.colDimension = numColumns;
    this.numberOfRows = numRows;
    this.matrix = new OpenIntObjectHashMap<>();
    this.nonZero = nonZero;
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
    this.nonZero = 0d;
    for (int i = 0; i < vectors.size(); i++) {
      this.matrix.put(i, vectors.get(i));
    }
  }

  public static void main(String[] args) {
    SparseMatrix m1 = new SparseMatrix(2, 3);
    m1.set(0, 1, -1);
    m1.set(0, 2, 2);
    m1.set(1, 0, 4);
    m1.set(1, 1, 11);
    m1.set(1, 2, 2);

    SparseMatrix m2 = new SparseMatrix(3, 2);
    m2.set(0, 0, 3);
    m2.set(0, 1, -1);
    m2.set(1, 0, 1);
    m2.set(0, 1, 2);
    m2.set(2, 0, 6);
    m2.set(2, 1, 1);

    Matrix m3 = m1.multiply(m2);
    System.out.println(m3);

  }

  @Override
  public Iterator<Entry> nonZeroIterator() {
    return new Iterator<Entry>() {
      private PrimitiveIterator.OfInt rowItr = IntStream.of(matrix.keys().toArray(new int[matrix.size()])).iterator();
      private int row;
      private Iterator<Vector.Entry> colItr;

      private boolean advance() {
        while (colItr == null || !colItr.hasNext()) {
          if (colItr == null && !rowItr.hasNext()) {
            return false;
          } else if (colItr == null) {
            row = rowItr.next();
            colItr = row(row).nonZeroIterator();
          } else if (!colItr.hasNext()) {
            colItr = null;
          } else {
            return true;
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
        Vector.Entry e = colItr.next();
        return new Matrix.Entry(row, e.index, e.value);
      }
    };
  }

  @Override
  public Iterator<Entry> orderedNonZeroIterator() {
    return new Iterator<Entry>() {
      private PrimitiveIterator.OfInt rowItr = IntStream.of(matrix.keys().toArray(new int[matrix.size()])).sorted().iterator();
      private int row;
      private Iterator<Vector.Entry> colItr;

      private boolean advance() {
        while (colItr == null || !colItr.hasNext()) {
          if (colItr == null && !rowItr.hasNext()) {
            return false;
          } else if (colItr == null) {
            row = rowItr.next();
            colItr = row(row).nonZeroIterator();
          } else if (!colItr.hasNext()) {
            colItr = null;
          } else {
            return true;
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
        Vector.Entry e = colItr.next();
        return new Matrix.Entry(row, e.index, e.value);
      }
    };
  }

  @Override
  public Vector column(int column) {
    Preconditions.checkArgument(column >= 0 && column < numberOfColumns());
    SparseVector col = new SparseVector(numberOfRows());
    for (int row : matrix.keys().elements()) {
      col.set(row, get(row, column));
    }
    return col;
  }

  @Override
  public Vector row(int row) {
    Preconditions.checkArgument(row >= 0 && row < numberOfRows());
    if (!matrix.containsKey(row)) {
      synchronized (matrix) {
        if (!matrix.containsKey(row)) {
          matrix.put(row, new SparseVector(numberOfColumns()));
        }
      }
    }
    return matrix.get(row);
  }

  @Override
  public double get(int row, int column) {
    return row(row).get(column);
  }

  @Override
  public void set(int row, int column, double value) {
    Vector r = row(row);
    r.set(column, value);
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
  public void setRow(int row, Vector vector) {
    Preconditions.checkNotNull(vector);
    Preconditions.checkArgument(row >= 0 && row < numberOfRows());
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
  public Matrix addSelf(Matrix other) {
    return null;
  }

  @Override
  public Matrix subtractSelf(Matrix other) {
    return null;
  }

  @Override
  public Matrix scaleSelf(double value) {
    for (int row : matrix.keys().elements()) {
      for (Vector.Entry entry : Collect.asIterable(row(row).nonZeroIterator())) {
        matrix.get(row).set(entry.index, entry.value * value);
      }
    }
    return this;
  }

  @Override
  public Matrix scale(double value) {
    Matrix m = new SparseMatrix(numberOfRows(), numberOfColumns());
    for (int row : matrix.keys().elements()) {
      for (Vector.Entry entry : Collect.asIterable(row(row).nonZeroIterator())) {
        m.set(row, entry.index, entry.value * value);
      }
    }
    return m;
  }

  @Override
  public Matrix incrementSelf(double value) {
    return null;
  }

  @Override
  public Matrix multiply(@NonNull Matrix m) {
    Preconditions.checkArgument(numberOfColumns() == m.numberOfRows(), "Dimension Mismatch");
    SparseMatrix mprime = new SparseMatrix(numberOfRows(), m.numberOfColumns());
    return mprime;
  }

  @Override
  public Matrix multiplySelf(@NonNull Matrix m) {
    Preconditions.checkArgument(numberOfColumns() == m.numberOfRows(), "Dimension Mismatch");
    IntStream.range(0, numberOfRows())
      .parallel()
      .forEach(r -> {
        setRow(r, row(r).multiplySelf(m.column(r)));
      });
    return this;
  }


  @Override
  public Matrix multiplySelf(Vector v) {
    return null;
  }

  @Override
  public Vector dot(Vector v) {
    SparseVector result = new SparseVector(numberOfRows);
    for (int i = 0; i < numberOfRows; i++) {
      result.set(i, row(i).dot(v));
    }
    return result;
  }

  @Override
  public Matrix transpose() {
    Matrix T = new SparseMatrix(numberOfColumns(), numberOfRows());
    for (int row : matrix.keys().elements()) {
      for (Vector.Entry entry : Collect.asIterable(row(row).nonZeroIterator())) {
        T.set(entry.index, row, entry.value);
      }
    }
    return T;
  }

  @Override
  public boolean isSparse() {
    return true;
  }

  @Override
  public Matrix increment(int row, int col, double amount) {
    matrix.get(row).increment(col, amount);
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
