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
import lombok.EqualsAndHashCode;
import lombok.NonNull;
import org.apache.commons.math3.linear.OpenMapRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

import java.io.Serializable;
import java.util.*;
import java.util.stream.IntStream;

/**
 * <p>Sparse implementation of Matrix stored in row-major order.</p>
 *
 * @author David B. Bracewell
 */
@EqualsAndHashCode(callSuper = false)
public class SparseMatrix implements Matrix, Serializable {
    private static final long serialVersionUID = -3802597548916836308L;
    final OpenMapRealMatrix matrix;

    /**
     * Instantiates a new Sparse matrix.
     *
     * @param numRows    the number of rows
     * @param numColumns the number of columns
     */
    public SparseMatrix(int numRows, int numColumns) {
        this.matrix = new OpenMapRealMatrix(numRows, numColumns);
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

    public SparseMatrix(@NonNull OpenMapRealMatrix matrix) {
        this.matrix = new OpenMapRealMatrix(matrix);
    }

    public SparseMatrix(@NonNull RealMatrix matrix) {
        this.matrix = new OpenMapRealMatrix(matrix.getRowDimension(), matrix.getColumnDimension());
        for (int r = 0; r < matrix.getRowDimension(); r++) {
            for (int c = 0; c < matrix.getColumnDimension(); c++) {
                this.matrix.setEntry(r, c, matrix.getEntry(r, c));
            }
        }
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
        int r = 0;
        int c = 0;
        if (vectors.size() > 0) {
            r = vectors.size();
            c = vectors
                    .get(0)
                    .dimension();
        }
        this.matrix = new OpenMapRealMatrix(r, c);
        for (int i = 0; i < vectors.size(); i++) {
            setRow(i, vectors.get(i));
        }
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
     * Random matrix.
     *
     * @param numberOfRows    the number of rows
     * @param numberOfColumns the number of columns
     * @return the matrix
     */
    public static Matrix random(int numberOfRows, int numberOfColumns) {
        Matrix m = new SparseMatrix(numberOfRows, numberOfColumns);
        IntStream
            .range(0, numberOfRows)
            .parallel()
            .forEach(r -> {
                for (int c = 0; c < numberOfColumns; c++) {
                    m.set(r, c, Math.random());
                }
            });
        return m;
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
     * Zeroes matrix.
     *
     * @param numberOfRows    the number of rows
     * @param numberOfColumns the number of columns
     * @return the matrix
     */
    public static Matrix zeroes(int numberOfRows, int numberOfColumns) {
        return new SparseMatrix(numberOfRows, numberOfColumns);
    }

    public RealMatrix asRealMatrix() {
        return matrix;
    }

    @Override
    public Matrix copy() {
        return new SparseMatrix(this);
    }

    private int encode(int row, int col) {
        return (row * numberOfColumns()) + col;
    }

    @Override
    public double get(int row, int column) {
        Preconditions.checkElementIndex(row, numberOfRows());
        Preconditions.checkElementIndex(column, numberOfColumns());
        return matrix.getEntry(row, column);
    }

    @Override
    public MatrixFactory getFactory() {
        return SparseMatrix::new;
    }

    @Override
    public Matrix increment(int row, int col, double amount) {
        Preconditions.checkElementIndex(row, numberOfRows());
        Preconditions.checkElementIndex(col, numberOfColumns());
        matrix.addToEntry(row, col, amount);
        return this;
    }

    @Override
    public boolean isSparse() {
        return true;
    }

    @Override
    public Iterator<Entry> nonZeroIterator() {
        return new Iterator<Entry>() {
            private PrimitiveIterator.OfInt rowItr = IntStream
                                                         .range(0, matrix.getRowDimension())
                                                         .iterator();
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
                        currentColumn = colItr
                                            .next()
                                            .getIndex();
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
    public int numberOfColumns() {
        return matrix.getColumnDimension();
    }

    @Override
    public int numberOfRows() {
        return matrix.getRowDimension();
    }

    @Override
    public Iterator<Entry> orderedNonZeroIterator() {
        return new Iterator<Entry>() {
            private PrimitiveIterator.OfInt rowItr = IntStream
                                                         .range(0, matrix.getRowDimension())
                                                         .iterator();
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
                        currentColumn = colItr
                                            .next()
                                            .getIndex();
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
    public Matrix set(int row, int column, double value) {
        Preconditions.checkElementIndex(row, numberOfRows());
        Preconditions.checkElementIndex(column, numberOfColumns());
        matrix.setEntry(row, column, value);
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
                builder
                    .append(get(row, col))
                    .append(", ");
            }
            builder.append("] ");
        }
        return builder
                   .append("]")
                   .toString();
    }

}//END OF SparseMatrix
