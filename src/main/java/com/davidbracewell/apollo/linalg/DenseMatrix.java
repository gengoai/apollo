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
  /**
   * The Matrix.
   */
  final DoubleMatrix matrix;

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

  private static DenseMatrix asJBLAS(Matrix m) {
    if (m instanceof DenseMatrix) {
      return (DenseMatrix) m;
    } else {
      return new DenseMatrix(m);
    }
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
  public Matrix add(Matrix m) {
    Preconditions.checkNotNull(m);
    return new DenseMatrix(matrix.add(asJBLAS(m).matrix));
  }

  @Override
  public Matrix subtract(Matrix m) {
    Preconditions.checkNotNull(m);
    return new DenseMatrix(matrix.sub(asJBLAS(m).matrix));

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
  public Matrix transpose() {
    return new DenseMatrix(matrix.transpose());
  }

  @Override
  public boolean isSparse() {
    return false;
  }

  @Override
  public Matrix addSelf(@NonNull Matrix other) {
    matrix.addi(asJBLAS(other).matrix);
    return this;
  }

  @Override
  public Matrix subtractSelf(@NonNull Matrix other) {
    matrix.subi(asJBLAS(other).matrix);
    return this;
  }

  @Override
  public Matrix multiplySelf(@NonNull Matrix other) {
    matrix.mmuli(asJBLAS(other).matrix);
    return this;
  }

  @Override
  public Matrix multiplySelf(Vector v) {
    return null;
  }

  @Override
  public Matrix copy() {
    return new DenseMatrix(toArray());
  }


  @Override
  public String toString() {
    return matrix.toString();
  }

}// END OF DenseMatrix
