package com.davidbracewell.apollo.linalg;

import org.jblas.DoubleMatrix;
import org.jblas.FloatMatrix;
import org.jblas.MatrixFunctions;
import org.jblas.ranges.IntervalRange;

import java.io.Serializable;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoublePredicate;
import java.util.function.DoubleUnaryOperator;

/**
 * @author David B. Bracewell
 */
public class DenseDoubleMatrix implements Matrix, Serializable {
   private static final long serialVersionUID = 1L;
   final DoubleMatrix matrix;

   public DenseDoubleMatrix(DoubleMatrix m) {
      this.matrix = m;
   }

   public DenseDoubleMatrix(int rows, int columns, double... values) {
      this.matrix = new DoubleMatrix(rows, columns, values);
   }

   public DenseDoubleMatrix(int len) {
      this.matrix = new DoubleMatrix(len);
   }

   public DenseDoubleMatrix(double[] values) {
      this.matrix = new DoubleMatrix(values);
   }

   public static DenseDoubleMatrix diag(Matrix matrix) {
      return wrap(DoubleMatrix.diag(matrix.toDoubleMatrix()));
   }

   public static DenseDoubleMatrix diag(Matrix matrix, int rows, int columns) {

      return wrap(DoubleMatrix.diag(matrix.toDoubleMatrix(), rows, columns));
   }

   public static DenseDoubleMatrix eye(int n) {
      return wrap(DoubleMatrix.eye(n));
   }

   public static DenseDoubleMatrix ones(int rows, int columns) {
      return wrap(DoubleMatrix.ones(rows, columns));
   }

   public static DenseDoubleMatrix ones(int length) {
      return wrap(DoubleMatrix.ones(length));
   }

   public static DenseDoubleMatrix rand(int rows, int columns) {
      return wrap(DoubleMatrix.rand(rows, columns));
   }

   public static DenseDoubleMatrix rand(int length) {
      return wrap(DoubleMatrix.rand(length));
   }

   public static DenseDoubleMatrix randn(int rows, int columns) {
      return wrap(DoubleMatrix.randn(rows, columns));
   }

   public static DenseDoubleMatrix randn(int length) {
      return wrap(DoubleMatrix.randn(length));
   }

   public static DenseDoubleMatrix scalar(double value) {
      return wrap(DoubleMatrix.scalar((float) value));
   }

   public static DenseDoubleMatrix wrap(DoubleMatrix fm) {
      return new DenseDoubleMatrix(fm);
   }

   public static DenseDoubleMatrix zeros(int rows, int columns) {
      return new DenseDoubleMatrix(rows, columns);
   }

   public static DenseDoubleMatrix zeros(int length) {
      return new DenseDoubleMatrix(length);
   }

   @Override
   public Matrix add(Matrix other) {
      return wrap(matrix.add(other.toDoubleMatrix()));
   }

   @Override
   public Matrix add(double scalar) {
      return wrap(matrix.add((float) scalar));
   }

   @Override
   public Matrix addColumnVector(Matrix columnVector) {
      return wrap(matrix.addColumnVector(columnVector.toDoubleMatrix()));
   }

   @Override
   public Matrix addRowVector(Matrix rowVector) {
      return wrap(matrix.addRowVector(rowVector.toDoubleMatrix()));
   }

   @Override
   public Matrix addi(Matrix other) {
      matrix.addi(other.toDoubleMatrix());
      return this;
   }

   @Override
   public Matrix addi(double scalar) {
      matrix.addi((float) scalar);
      return this;
   }

   @Override
   public Matrix addiColumnVector(Matrix columnVector) {
      matrix.addiColumnVector(columnVector.toDoubleMatrix());
      return this;
   }

   @Override
   public Matrix addiRowVector(Matrix rowVector) {
      matrix.addiRowVector(rowVector.toDoubleMatrix());
      return this;
   }

   @Override
   public int[] columnArgMaxs() {
      return matrix.columnArgmaxs();
   }

   @Override
   public int[] columnArgMins() {
      return matrix.columnArgmins();
   }

   @Override
   public Matrix columnMaxs() {
      return wrap(matrix.columnMaxs());
   }

   @Override
   public Matrix columnMeans() {
      return wrap(matrix.rowMeans());
   }

   @Override
   public Matrix columnMins() {
      return wrap(matrix.columnMins());
   }

   @Override
   public Matrix columnSums() {
      return wrap(matrix.columnSums());
   }

   @Override
   public Matrix copy() {
      return wrap(matrix.dup());
   }

   @Override
   public Matrix div(Matrix other) {
      return wrap(matrix.div(other.toDoubleMatrix()));
   }

   @Override
   public Matrix div(double scalar) {
      return wrap(matrix.div((float) scalar));
   }

   @Override
   public Matrix divColumnVector(Matrix columnVector) {
      return wrap(matrix.divColumnVector(columnVector.toDoubleMatrix()));
   }

   @Override
   public Matrix divRowVector(Matrix rowVector) {
      return wrap(matrix.divRowVector(rowVector.toDoubleMatrix()));
   }

   @Override
   public Matrix divi(Matrix other) {
      matrix.divi(other.toDoubleMatrix());
      return this;
   }

   @Override
   public Matrix divi(double scalar) {
      matrix.divi((float) scalar);
      return this;
   }

   @Override
   public Matrix diviColumnVector(Matrix columnVector) {
      matrix.diviColumnVector(columnVector.toDoubleMatrix());
      return this;
   }

   @Override
   public Matrix diviRowVector(Matrix rowVector) {
      matrix.diviRowVector(rowVector.toDoubleMatrix());
      return this;
   }

   @Override
   public double dot(Matrix other) {
      return matrix.dot(other.toDoubleMatrix());
   }

   @Override
   public Matrix exp() {
      return wrap(MatrixFunctions.exp(matrix));
   }

   @Override
   public double get(int r, int c) {
      return matrix.get(r, c);
   }

   @Override
   public double get(int i) {
      return matrix.get(i);
   }

   @Override
   public Matrix getColumn(int column) {
      return wrap(matrix.getColumn(column));
   }

   @Override
   public Matrix getColumns(int[] cindexes) {
      return wrap(matrix.getColumns(cindexes));
   }

   @Override
   public Matrix getColumns(int from, int to) {
      return wrap(matrix.getColumns(new IntervalRange(from, to)));
   }

   @Override
   public ElementType getElementType() {
      return ElementType.FLOAT;
   }

   @Override
   public MatrixFactory getFactory() {
      return MatrixFactory.DENSE_DOUBLE;
   }

   @Override
   public Matrix getRow(int row) {
      return wrap(matrix.getRow(row));
   }

   @Override
   public Matrix getRows(int[] rindexes) {
      return wrap(matrix.getRows(rindexes));
   }

   @Override
   public Matrix getRows(int from, int to) {
      return wrap(matrix.getRows(new IntervalRange(from, to)));
   }

   @Override
   public boolean isColumnVector() {
      return matrix.isColumnVector();
   }

   @Override
   public boolean isEmpty() {
      return matrix.isEmpty();
   }

   @Override
   public Matrix isInfinite() {
      return wrap(matrix.isInfinite());
   }

   @Override
   public Matrix isInfinitei() {
      matrix.isInfinitei();
      return this;
   }

   @Override
   public boolean isLowerTriangular() {
      return matrix.isLowerTriangular();
   }

   @Override
   public Matrix isNaN() {
      return wrap(matrix.isNaN());
   }

   @Override
   public Matrix isNaNi() {
      matrix.isNaNi();
      return this;
   }

   @Override
   public boolean isRowVector() {
      return matrix.isRowVector();
   }

   @Override
   public boolean isScalar() {
      return matrix.isSquare();
   }

   @Override
   public boolean isSquare() {
      return matrix.isSquare();
   }

   @Override
   public boolean isUpperTriangular() {
      return matrix.isUpperTriangular();
   }

   @Override
   public boolean isVector() {
      return matrix.isVector();
   }

   @Override
   public Matrix log() {
      return wrap(MatrixFunctions.log(matrix));
   }

   @Override
   public Matrix map(DoubleUnaryOperator operator) {
      double[] out = new double[matrix.length];
      for (int i = 0; i < out.length; i++) {
         out[i] = (float) operator.applyAsDouble(matrix.data[i]);
      }
      return new DenseDoubleMatrix(matrix.rows, matrix.columns, out);
   }

   @Override
   public Matrix map(DoubleBinaryOperator operator, Matrix other) {
      double[] out = new double[matrix.length];
      DoubleMatrix om = other.toDoubleMatrix();
      for (int i = 0; i < out.length; i++) {
         out[i] = (float) operator.applyAsDouble(matrix.data[i], om.data[i]);
      }
      return new DenseDoubleMatrix(matrix.rows, matrix.columns, out);
   }

   @Override
   public Matrix mapi(DoubleUnaryOperator operator) {
      for (int i = 0; i < matrix.data.length; i++) {
         matrix.data[i] = (float) operator.applyAsDouble(matrix.data[i]);
      }
      return this;
   }

   @Override
   public Matrix mapi(DoubleBinaryOperator operator, Matrix other) {
      DoubleMatrix om = other.toDoubleMatrix();
      for (int i = 0; i < matrix.data.length; i++) {
         matrix.data[i] = (float) operator.applyAsDouble(matrix.data[i], om.data[i]);
      }
      return this;
   }

   @Override
   public Matrix max(double scalar) {
      return wrap(matrix.max((float) scalar));
   }

   @Override
   public Matrix max(Matrix other) {
      return wrap(matrix.max(other.toDoubleMatrix()));
   }

   @Override
   public Matrix maxi(double scalar) {
      matrix.maxi((float) scalar);
      return this;
   }

   @Override
   public Matrix maxi(Matrix other) {
      matrix.maxi(other.toDoubleMatrix());
      return this;
   }

   @Override
   public double mean() {
      return matrix.mean();
   }

   @Override
   public Matrix mmul(Matrix other) {
      return wrap(matrix.mmul(other.toDoubleMatrix()));
   }

   @Override
   public Matrix mmuli(Matrix other) {
      matrix.mmuli(other.toDoubleMatrix());
      return this;
   }

   @Override
   public Matrix mul(Matrix other) {
      return wrap(matrix.mul(other.toDoubleMatrix()));
   }

   @Override
   public Matrix mul(double scalar) {
      return wrap(matrix.mul((float) scalar));
   }

   @Override
   public Matrix mulColumnVector(Matrix columnVector) {
      return wrap(matrix.mulColumnVector(columnVector.toDoubleMatrix()));
   }

   @Override
   public Matrix mulRowVector(Matrix rowVector) {
      return wrap(matrix.mulRowVector(rowVector.toDoubleMatrix()));
   }

   @Override
   public Matrix muli(Matrix other) {
      matrix.muli(other.toDoubleMatrix());
      return this;
   }

   @Override
   public Matrix muli(double scalar) {
      matrix.muli((float) scalar);
      return this;
   }

   @Override
   public Matrix muliColumnVector(Matrix columnVector) {
      matrix.muliColumnVector(columnVector.toDoubleMatrix());
      return this;
   }

   @Override
   public Matrix muliRowVector(Matrix rowVector) {
      matrix.muliRowVector(rowVector.toDoubleMatrix());
      return this;
   }

   @Override
   public Matrix neg() {
      return wrap(matrix.neg());
   }

   @Override
   public Matrix negi() {
      matrix.negi();
      return this;
   }

   @Override
   public int numCols() {
      return matrix.columns;
   }

   @Override
   public int numRows() {
      return matrix.rows;
   }

   @Override
   public Matrix predicate(DoublePredicate predicate) {
      double[] out = new double[matrix.length];
      for (int i = 0; i < out.length; i++) {
         out[i] = predicate.test(matrix.data[i]) ? 1.0f : 0.0f;
      }
      return new DenseDoubleMatrix(matrix.rows, matrix.columns, out);
   }

   @Override
   public Matrix predicatei(DoublePredicate predicate) {
      for (int i = 0; i < matrix.data.length; i++) {
         matrix.data[i] = predicate.test(matrix.data[i]) ? 1.0f : 0.0f;
      }
      return this;
   }

   @Override
   public Matrix rdiv(double scalar) {
      return wrap(matrix.rdiv((float) scalar));
   }

   @Override
   public Matrix rdivi(Matrix other) {
      return wrap(matrix.rdiv(other.toDoubleMatrix()));
   }

   @Override
   public Matrix rdivi(double scalar) {
      matrix.rdivi((float) scalar);
      return this;
   }

   @Override
   public int[] rowArgMaxs() {
      return matrix.rowArgmaxs();
   }

   @Override
   public int[] rowArgMins() {
      return matrix.rowArgmins();
   }

   @Override
   public Matrix rowMaxs() {
      return wrap(matrix.rowMaxs());
   }

   @Override
   public Matrix rowMeans() {
      return wrap(matrix.rowMeans());
   }

   @Override
   public Matrix rowMins() {
      return wrap(matrix.rowMins());
   }

   @Override
   public Matrix rowSums() {
      return wrap(matrix.rowSums());
   }

   @Override
   public Matrix rsub(double scalar) {
      return wrap(matrix.rsub((float) scalar));
   }

   @Override
   public Matrix rsubi(Matrix other) {
      matrix.rsubi(other.toDoubleMatrix());
      return this;
   }

   @Override
   public Matrix rsubi(double scalar) {
      matrix.rsubi((float) scalar);
      return this;
   }

   @Override
   public Matrix select(Matrix where) {
      return wrap(matrix.select(where.toDoubleMatrix()));
   }

   @Override
   public Matrix selecti(Matrix where) {
      matrix.selecti(where.toDoubleMatrix());
      return this;
   }

   @Override
   public Matrix set(int r, int c, double value) {
      matrix.put(r, c, (float) value);
      return this;
   }

   @Override
   public Matrix set(int index, double value) {
      matrix.put(index, (float) value);
      return this;
   }

   @Override
   public Matrix reshape(int rows, int columns) {
      matrix.reshape(rows, columns);
      return this;
   }

   @Override
   public Matrix resize(int rows, int columns) {
      matrix.resize(rows, columns);
      return this;
   }

   @Override
   public Matrix setColumn(int c, Matrix columnVector) {
      matrix.putColumn(c, columnVector.toDoubleMatrix());
      return this;
   }

   @Override
   public int length() {
      return matrix.length;
   }

   @Override
   public Matrix setRow(int r, Matrix rowVector) {
      matrix.putRow(r, rowVector.toDoubleMatrix());
      return this;
   }
   @Override
   public Matrix sub(Matrix other) {
      return wrap(matrix.sub(other.toDoubleMatrix()));
   }

   @Override
   public Matrix sub(double scalar) {
      return wrap(matrix.sub((float) scalar));
   }

   @Override
   public Matrix subColumnVector(Matrix columnVector) {
      return wrap(matrix.subColumnVector(columnVector.toDoubleMatrix()));
   }

   @Override
   public Matrix subRowVector(Matrix rowVector) {
      return wrap(matrix.subRowVector(rowVector.toDoubleMatrix()));
   }

   @Override
   public Matrix subi(Matrix other) {
      matrix.subi(other.toDoubleMatrix());
      return this;
   }

   @Override
   public Matrix subi(double scalar) {
      matrix.subi((float) scalar);
      return this;
   }

   @Override
   public Matrix subiColumnVector(Matrix columnVector) {
      matrix.subiColumnVector(columnVector.toDoubleMatrix());
      return this;
   }

   @Override
   public Matrix subiRowVector(Matrix rowVector) {
      matrix.subiRowVector(rowVector.toDoubleMatrix());
      return this;
   }

   @Override
   public double sum() {
      return matrix.sum();
   }

   @Override
   public double[] toDoubleArray() {
      return matrix.toArray();
   }

   @Override
   public DoubleMatrix toDoubleMatrix() {
      return matrix;
   }

   @Override
   public float[] toFloatArray() {
      float[] out = new float[matrix.length];
      for (int i = 0; i < out.length; i++) {
         out[i] = (float) matrix.data[i];
      }
      return out;
   }

   @Override
   public FloatMatrix toFloatMatrix() {
      return MatrixFunctions.doubleToFloat(matrix);
   }

   @Override
   public Matrix transpose() {
      return wrap(matrix.transpose());
   }
}// END OF DenseDoubleMatrix
