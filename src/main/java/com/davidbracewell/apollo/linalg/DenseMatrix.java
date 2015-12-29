package com.davidbracewell.apollo.linalg;

import com.davidbracewell.collection.Collect;
import com.google.common.base.Preconditions;
import lombok.NonNull;
import org.jblas.ComplexDouble;
import org.jblas.DoubleMatrix;

import java.io.Serializable;

/**
 * The type Dense matrix.
 *
 * @author David B. Bracewell
 */
public class DenseMatrix implements Matrix, Serializable {
  private static final long serialVersionUID = 1L;
  /**
   * The Matrix.
   */
  volatile DoubleMatrix matrix;

  /**
   * Instantiates a new Dense matrix.
   *
   * @param rowDimension    the row dimension
   * @param columnDimension the column dimension
   */
  public DenseMatrix(int rowDimension, int columnDimension) {
    this.matrix = new DoubleMatrix(rowDimension, columnDimension);
  }

  /**
   * Instantiates a new Dense matrix.
   *
   * @param matrix the matrix
   */
  public DenseMatrix(double[][] matrix) {
    this.matrix = new DoubleMatrix(matrix);
  }

  public DenseMatrix(@NonNull ComplexDouble[][] matrix) {
    this.matrix = new DoubleMatrix(matrix.length, matrix[0].length);
    for (int r = 0; r < matrix.length; r++) {
      for (int c = 0; c < matrix[c].length; c++) {
        this.matrix.put(r, c, matrix[r][c].real());
      }
    }
  }

  /**
   * Instantiates a new Dense matrix.
   *
   * @param matrix the matrix
   */
  public DenseMatrix(@NonNull Matrix matrix) {
    this.matrix = new DoubleMatrix(matrix.numberOfRows(), matrix.numberOfColumns());
    matrix.forEachSparse(e -> this.matrix.put(e.row, e.column, e.value));
  }

  /**
   * Instantiates a new Dense matrix.
   *
   * @param matrix the matrix
   */
  public DenseMatrix(DoubleMatrix matrix) {
    this.matrix = matrix;
  }

  /**
   * Zeroes matrix.
   *
   * @param rowDimension    the row dimension
   * @param columnDimension the column dimension
   * @return the matrix
   */
  public static Matrix zeroes(int rowDimension, int columnDimension) {
    return new DenseMatrix(DoubleMatrix.zeros(rowDimension, columnDimension));
  }

  /**
   * Ones matrix.
   *
   * @param rowDimension    the row dimension
   * @param columnDimension the column dimension
   * @return the matrix
   */
  public static Matrix ones(int rowDimension, int columnDimension) {
    return new DenseMatrix(DoubleMatrix.ones(rowDimension, columnDimension));
  }

  /**
   * Unit matrix.
   *
   * @param N the n
   * @return the matrix
   */
  public static Matrix unit(int N) {
    return new DenseMatrix(DoubleMatrix.eye(N));
  }

  public DoubleMatrix asDoubleMatrix() {
    return matrix;
  }

  @Override
  public Vector dot(Vector v) {
    DenseVector result = new DenseVector(numberOfRows());
    for (int i = 0; i < numberOfRows(); i++) {
      result.set(i, row(i).dot(v));
    }
    return result;
  }


  @Override
  public Matrix increment(int row, int col, double amount) {
    set(row, col, get(row, col) + amount);
    return this;
  }

  @Override
  public Vector column(int column) {
    return new DenseVector(matrix.getColumn(column).toArray());
  }

  @Override
  public Vector row(int row) {
    return new DenseVector(matrix.getRow(row).toArray());
  }

  @Override
  public double get(int row, int column) {
    return matrix.get(row, column);
  }

  @Override
  public void set(int row, int column, double value) {
    matrix.put(row, column, value);
  }

  @Override
  public void setColumn(int column, Vector vector) {
    for (Vector.Entry entry : Collect.asIterable(vector.nonZeroIterator())) {
      set(entry.index, column, entry.value);
    }
  }

  @Override
  public void setRow(int row, Vector vector) {
    for (Vector.Entry entry : Collect.asIterable(vector.iterator())) {
      set(row, entry.index, entry.value);
    }
  }

  @Override
  public double[][] toArray() {
    return matrix.toArray2();
  }

  @Override
  public int numberOfRows() {
    return matrix.rows;
  }

  @Override
  public int numberOfColumns() {
    return matrix.columns;
  }

  @Override
  public Matrix add(@NonNull Matrix m) {
    if (m instanceof DenseMatrix) {
      return new DenseMatrix(matrix.add(m.toDense().matrix));
    }
    return Matrix.super.add(m);
  }

  @Override
  public Matrix subtract(@NonNull Matrix m) {
    if (m instanceof DenseMatrix) {
      return new DenseMatrix(matrix.sub(m.toDense().matrix));
    }
    return Matrix.super.add(m);
  }

  @Override
  public Matrix scaleSelf(double value) {
    matrix.muli(value);
    return this;
  }

  @Override
  public Matrix incrementSelf(double value) {
    matrix.addi(value);
    return this;
  }

  @Override
  public Matrix multiply(@NonNull Matrix m) {
    Preconditions.checkArgument(m.numberOfColumns() == numberOfRows(), "Dimension Mismatch");
    return new DenseMatrix(matrix.mmul(m.toDense().matrix));
  }


  @Override
  public Matrix transpose() {
    return new DenseMatrix(matrix.transpose());
  }

  @Override
  public boolean isSparse() {
    return false;
  }

  @Override
  public Matrix addSelf(@NonNull Matrix other) {
    Preconditions.checkArgument(other.numberOfColumns() == numberOfColumns() && other.numberOfRows() == numberOfRows(), "Dimension Mismatch");
    if (other instanceof DenseMatrix) {
      matrix.addi(other.toDense().matrix);
    } else {
      other.forEachSparse(e -> increment(e.row, e.column, e.value));
    }
    return this;
  }

  @Override
  public Matrix subtractSelf(@NonNull Matrix other) {
    Preconditions.checkArgument(other.numberOfColumns() == numberOfColumns() && other.numberOfRows() == numberOfRows(), "Dimension Mismatch");
    if (other instanceof DenseMatrix) {
      matrix.subi(other.toDense().matrix);
    } else {
      other.forEachSparse(e -> increment(e.row, e.column, e.value));
    }
    return this;
  }

  @Override
  public Matrix scaleSelf(@NonNull Matrix other) {
    Preconditions.checkArgument(other.numberOfColumns() == numberOfColumns() && other.numberOfRows() == numberOfRows(), "Dimension Mismatch");
    if (other instanceof DenseMatrix) {
      matrix.muli(other.toDense().matrix);
    } else {
      forEachSparse(e -> increment(e.row, e.column, other.get(e.row, e.column)));
    }
    return this;
  }

  @Override
  public Matrix scaleSelf(@NonNull Vector other) {
    Preconditions.checkArgument(other.dimension() == numberOfColumns(), "Dimension Mismatch");
    for (int r = 0; r < numberOfRows(); r++) {
      for (int c = 0; c < numberOfColumns(); c++) {
        set(r, c, get(r, c) * other.get(c));
      }
    }
    return this;
  }

  @Override
  public Matrix copy() {
    return new DenseMatrix(toArray());
  }


  @Override
  public String toString() {
    return matrix.toString();
  }

  @Override
  public DenseMatrix toDense() {
    return this;
  }

}// END OF DenseMatrix
