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
 *
 */

package com.gengoai.apollo.math.linalg;

import com.gengoai.Validation;
import com.gengoai.conversion.Cast;
import lombok.NonNull;
import org.jblas.DoubleMatrix;
import org.jblas.ranges.IntervalRange;

import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;

/**
 * Matrix implementation based on JBlas
 *
 * @author David B. Bracewell
 */
public class DenseMatrix extends Matrix {
   private static final long serialVersionUID = 1L;
   private final DoubleMatrix matrix;

   /**
    * Instantiates a new Dense Matrix.
    *
    * @param dims the dimensions
    */
   public DenseMatrix(@NonNull int... dims) {
      this(new Shape(dims));
   }

   /**
    * Instantiates a new Dense Matrix.
    *
    * @param shape the shape
    */
   public DenseMatrix(@NonNull Shape shape) {
      super(shape);
      Validation.checkArgument(shape.order() < 3, () -> "Invalid Shape: " + shape);
      this.matrix = new DoubleMatrix(shape.rows(), shape.columns());
   }

   /**
    * Instantiates a new Dense Matrix from a DoubleMatrix
    *
    * @param doubleMatrix the double matrix
    */
   public DenseMatrix(@NonNull DoubleMatrix doubleMatrix) {
      super(new Shape(doubleMatrix.rows, doubleMatrix.columns));
      this.matrix = doubleMatrix;
   }

   @Override
   public NDArray T() {
      return new DenseMatrix(matrix.transpose());
   }

   @Override
   public NDArray add(@NonNull NDArray rhs) {
      if(rhs.isDense()) {
         checkLength(rhs.shape);
         return new DenseMatrix(matrix.add(rhs.toDoubleMatrix()[0]));
      }
      return super.add(rhs);
   }

   @Override
   public NDArray addi(@NonNull NDArray rhs) {
      if(rhs.isDense()) {
         checkLength(rhs.shape);
         matrix.addi(rhs.toDoubleMatrix()[0]);
         return this;
      }
      return super.add(rhs);
   }

   @Override
   public NDArray addiColumnVector(@NonNull NDArray rhs) {
      if(rhs.isDense()) {
         matrix.addiColumnVector(rhs.toDoubleMatrix()[0]);
         return this;
      }
      return super.addiColumnVector(rhs);
   }

   @Override
   public NDArray addiRowVector(@NonNull NDArray rhs) {
      if(rhs.isDense()) {
         matrix.addiRowVector(rhs.toDoubleMatrix()[0]);
         return this;
      }
      return super.addiRowVector(rhs);
   }

   @Override
   public long argmax() {
      if(matrix.argmax() < 0) {
         System.out.println(matrix);
      }
      return matrix.argmax();
   }

   @Override
   public long argmin() {
      return matrix.argmin();
   }

   @Override
   public NDArray columnSums() {
      return new DenseMatrix(matrix.columnSums());
   }

   @Override
   public NDArray compact() {
      return this;
   }

   @Override
   public NDArray div(@NonNull NDArray rhs) {
      if(rhs.isDense()) {
         return new DenseMatrix(matrix.div(rhs.toDoubleMatrix()[0]));
      }
      return super.div(rhs);
   }

   @Override
   public NDArray divi(@NonNull NDArray rhs) {
      if(rhs.isDense()) {
         matrix.divi(rhs.toDoubleMatrix()[0]);
         return this;
      }
      return super.divi(rhs);
   }

   @Override
   public NDArray diviColumnVector(@NonNull NDArray rhs) {
      if(rhs.isDense()) {
         matrix.diviColumnVector(rhs.toDoubleMatrix()[0]);
         return this;
      }
      return super.diviColumnVector(rhs);
   }

   @Override
   public NDArray diviRowVector(@NonNull NDArray rhs) {
      if(rhs.isDense()) {
         matrix.diviRowVector(rhs.toDoubleMatrix()[0]);
         return this;
      }
      return super.diviRowVector(rhs);
   }

   @Override
   public double dot(@NonNull NDArray rhs) {
      if(rhs.isDense()) {
         checkLength(rhs.shape());
         return matrix.dot(rhs.toDoubleMatrix()[0]);
      } else {
         return rhs.dot(this);
      }
   }

   @Override
   public NDArray fill(double value) {
      matrix.fill(value);
      return this;
   }

   @Override
   public void forEachSparse(@NonNull EntryConsumer consumer) {
      for(int i = 0; i < matrix.length; i++) {
         consumer.apply(i, matrix.data[i]);
      }
   }

   @Override
   public double get(long i) {
      return matrix.get((int) i);
   }

   @Override
   public double get(int row, int col) {
      return matrix.get(row, col);
   }

   @Override
   public NDArray getColumn(int column) {
      return new DenseMatrix(matrix.getColumn(column));
   }

   @Override
   public NDArray getColumns(int[] columns) {
      return new DenseMatrix(matrix.getColumns(columns));
   }

   @Override
   public NDArray getRow(int row) {
      return new DenseMatrix(matrix.getRow(row));
   }

   @Override
   public NDArray getRows(int[] rows) {
      return new DenseMatrix(matrix.getRows(rows));
   }

   @Override
   public NDArray getSubMatrix(int fromRow, int toRow, int fromCol, int toCol) {
      return new DenseMatrix(matrix.get(new IntervalRange(fromRow, toRow),
                                        new IntervalRange(fromCol, toCol)));
   }

   @Override
   public boolean isDense() {
      return true;
   }

   @Override
   public NDArray map(@NonNull NDArray rhs, @NonNull DoubleBinaryOperator operator) {
      DenseMatrix out = Cast.as(zeroLike());
      for(int i = 0; i < shape.matrixLength; i++) {
         out.matrix.data[i] = operator.applyAsDouble(matrix.data[i], rhs.get(i));
      }
      return out;
   }

   @Override
   public NDArray map(@NonNull DoubleUnaryOperator operator) {
      DenseMatrix dm = new DenseMatrix(matrix.rows, matrix.columns);
      for(int i = 0; i < matrix.length; i++) {
         dm.matrix.data[i] = operator.applyAsDouble(matrix.data[i]);
      }
      return dm;
   }

   @Override
   public NDArray mapi(@NonNull NDArray rhs, @NonNull DoubleBinaryOperator operator) {
      checkLength(rhs.shape());
      for(int i = 0; i < shape.matrixLength; i++) {
         matrix.data[i] = operator.applyAsDouble(matrix.data[i], rhs.get(i));
      }
      return this;
   }

   @Override
   public NDArray mapi(@NonNull DoubleUnaryOperator operator) {
      for(int i = 0; i < matrix.length; i++) {
         matrix.data[i] = operator.applyAsDouble(matrix.data[i]);
      }
      return this;
   }

   @Override
   public double max() {
      return matrix.max();
   }

   @Override
   public double min() {
      return matrix.min();
   }

   @Override
   public NDArray mmul(@NonNull NDArray rhs) {
      Validation.checkArgument(rhs.shape.sliceLength == 1,
                               () -> "Invalid Slice Length: " + rhs.shape.sliceLength + " != 1");
      if(shape.isVector()) {
         super.mmul(rhs);
      }
      return new DenseMatrix(matrix.mmul(rhs.toDoubleMatrix()[0]));
   }

   @Override
   public NDArray mul(@NonNull NDArray rhs) {
      if(rhs.isDense()) {
         checkLength(rhs.shape());
         return new DenseMatrix(matrix.mul(rhs.toDoubleMatrix()[0]));
      }
      return super.mul(rhs);
   }

   @Override
   public NDArray muli(@NonNull NDArray rhs) {
      if(rhs.isDense()) {
         checkLength(rhs.shape());
         matrix.muli(rhs.toDoubleMatrix()[0]);
         return this;
      }
      return super.muli(rhs);
   }

   @Override
   public NDArray muliColumnVector(@NonNull NDArray rhs) {
      if(rhs.isDense()) {
         checkLength(rows(), rhs.shape());
         matrix.muliColumnVector(rhs.toDoubleMatrix()[0]);
         return this;
      }
      return super.muliColumnVector(rhs);
   }

   @Override
   public NDArray muliRowVector(@NonNull NDArray rhs) {
      if(rhs.isDense()) {
         checkLength(columns(), rhs.shape());
         matrix.muliRowVector(rhs.toDoubleMatrix()[0]);
         return this;
      }
      return super.muliRowVector(rhs);
   }

   @Override
   public NDArray rdiv(@NonNull NDArray rhs) {
      checkLength(rhs.shape());
      if(rhs.isDense()) {
         return new DenseMatrix(matrix.rdiv(rhs.toDoubleMatrix()[0]));
      }
      return super.rdiv(rhs);
   }

   @Override
   public NDArray rdivi(@NonNull NDArray rhs) {
      checkLength(rhs.shape());
      if(rhs.isDense()) {
         matrix.rdivi(rhs.toDoubleMatrix()[0]);
         return this;
      }
      return super.rdivi(rhs);
   }

   @Override
   public NDArray reshape(@NonNull int... dims) {
      shape.reshape(dims);
      matrix.reshape(shape.rows(), shape.columns());
      return this;
   }

   @Override
   public NDArray rowSums() {
      return new DenseMatrix(matrix.rowSums());
   }

   @Override
   public NDArray rsub(@NonNull NDArray rhs) {
      checkLength(rhs.shape());
      if(rhs.isDense()) {
         return new DenseMatrix(matrix.rsub(rhs.toDoubleMatrix()[0]));
      }
      return super.rsub(rhs);
   }

   @Override
   public NDArray rsubi(@NonNull NDArray rhs) {
      checkLength(rhs.shape());
      if(rhs.isDense()) {
         matrix.rsubi(rhs.toDoubleMatrix()[0]);
         return this;
      }
      return super.rsubi(rhs);
   }

   @Override
   public NDArray set(long i, double value) {
      matrix.put((int) i, value);
      return this;
   }

   @Override
   public NDArray set(int row, int col, double value) {
      matrix.put(row, col, value);
      return this;
   }

   @Override
   public NDArray setColumn(int column, @NonNull NDArray array) {
      checkLength(shape.rows(), array.shape());
      matrix.putColumn(column, array.toDoubleMatrix()[0]);
      return this;
   }

   @Override
   public NDArray setRow(int row, @NonNull NDArray array) {
      checkLength(shape.columns(), array.shape());
      matrix.putRow(row, array.toDoubleMatrix()[0]);
      return this;
   }

   @Override
   public NDArray sub(@NonNull NDArray rhs) {
      checkLength(rhs.shape());
      if(rhs.isDense()) {
         return new DenseMatrix(matrix.sub(rhs.toDoubleMatrix()[0]));
      }
      return super.sub(rhs);
   }

   @Override
   public NDArray subi(@NonNull NDArray rhs) {
      checkLength(rhs.shape());
      if(rhs.isDense()) {
         matrix.subi(rhs.toDoubleMatrix()[0]);
         return this;
      }
      return super.subi(rhs);
   }

   @Override
   public NDArray subiColumnVector(@NonNull NDArray rhs) {
      if(rhs.isDense()) {
         matrix.subiColumnVector(rhs.toDoubleMatrix()[0]);
         return this;
      }
      return super.subiColumnVector(rhs);
   }

   @Override
   public NDArray subiRowVector(@NonNull NDArray rhs) {
      if(rhs.isDense()) {
         matrix.subiRowVector(rhs.toDoubleMatrix()[0]);
         return this;
      }
      return super.subiRowVector(rhs);
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
   public DoubleMatrix[] toDoubleMatrix() {
      return new DoubleMatrix[]{matrix};
   }

   @Override
   public NDArray zeroLike() {
      return new DenseMatrix(shape);
   }
}//END OF DenseTwoDArray
