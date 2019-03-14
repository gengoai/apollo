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

package com.gengoai.apollo.linear;

import com.gengoai.conversion.Cast;
import org.jblas.DoubleMatrix;
import org.jblas.ranges.IntervalRange;

import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;

/**
 * @author David B. Bracewell
 */
public class DenseMatrix extends Matrix {
   private final DoubleMatrix matrix;

   public DenseMatrix(int... dims) {
      this(new Shape(dims));
   }

   public DenseMatrix(Shape shape) {
      super(shape);
      this.matrix = new DoubleMatrix(shape.rows(), shape.columns());
   }

   public DenseMatrix(DoubleMatrix fm) {
      super(new Shape(fm.rows, fm.columns));
      this.matrix = fm;
   }

   public static void loop() {
      NDArray n1 = NDArrayFactory.DENSE.rand(1000, 1000);
      NDArray n4 = NDArrayFactory.SPARSE.rand(1000, 1000);
      for (int i = 0; i < 10_000; i++) {
         n1.dot(n4);
      }
   }

   @Override
   public NDArray getSubMatrix(int fromRow, int toRow, int fromCol, int toCol) {
      return new DenseMatrix(matrix.get(new IntervalRange(fromRow, toRow),
                                        new IntervalRange(fromCol, toCol)));
   }

   @Override
   public double dot(NDArray rhs) {
      if (rhs.isDense()) {
         checkLength(shape, rhs.shape());
         return matrix.dot(rhs.toDoubleMatrix()[0]);
      } else {
         return rhs.dot(this);
      }
   }

   public static void main(String[] args) throws Exception {
//      loop();
//      System.gc();
//      Threads.sleep(10_000);
//      Stopwatch sw = Stopwatch.createStarted();
//      loop();
//      System.out.println(sw);

      DoubleMatrix m = new DoubleMatrix(4, 4);
   }

   @Override
   public NDArray mmul(NDArray rhs) {
      return new DenseMatrix(matrix.mmul(rhs.toDoubleMatrix()[0]));
   }

   @Override
   public NDArray T() {
      return new DenseMatrix(matrix.transpose());
   }

   @Override
   public NDArray add(NDArray rhs) {
      if (rhs.isDense()) {
         return new DenseMatrix(matrix.add(rhs.toDoubleMatrix()[0]));
      }
      return super.add(rhs);
   }

   @Override
   public NDArray addi(NDArray rhs) {
      if (rhs.isDense()) {
         matrix.addi(rhs.toDoubleMatrix()[0]);
         return this;
      }
      return super.add(rhs);
   }

   @Override
   public NDArray addiColumnVector(NDArray rhs) {
      if (rhs.isDense()) {
         matrix.addiColumnVector(rhs.toDoubleMatrix()[0]);
         return this;
      }
      return super.addiColumnVector(rhs);
   }

   @Override
   public double[] toDoubleArray() {
      return matrix.toArray();
   }

   @Override
   public void forEachSparse(EntryConsumer consumer) {
      for (int i = 0; i < matrix.length; i++) {
         consumer.apply(i, matrix.data[i]);
      }
   }

   @Override
   public NDArray addiRowVector(NDArray rhs) {
      if (rhs.isDense()) {
         matrix.addiRowVector(rhs.toDoubleMatrix()[0]);
         return this;
      }
      return super.addiRowVector(rhs);
   }

   @Override
   public NDArray copy() {
      DenseMatrix dm = new DenseMatrix(matrix.dup());


      return dm;
   }

   @Override
   public NDArray div(NDArray rhs) {
      if (rhs.isDense()) {
         return new DenseMatrix(matrix.div(rhs.toDoubleMatrix()[0]));
      }
      return super.div(rhs);
   }

   @Override
   public NDArray divi(NDArray rhs) {
      if (rhs.isDense()) {
         matrix.divi(rhs.toDoubleMatrix()[0]);
         return this;
      }
      return super.divi(rhs);
   }

   @Override
   public NDArray diviColumnVector(NDArray rhs) {
      if (rhs.isDense()) {
         matrix.diviColumnVector(rhs.toDoubleMatrix()[0]);
         return this;
      }
      return super.diviColumnVector(rhs);
   }

   @Override
   public NDArray diviRowVector(NDArray rhs) {
      if (rhs.isDense()) {
         matrix.diviRowVector(rhs.toDoubleMatrix()[0]);
         return this;
      }
      return super.diviRowVector(rhs);
   }

   @Override
   public NDArray fill(double value) {
      matrix.fill(value);
      return this;
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
   public boolean isDense() {
      return true;
   }

   @Override
   public NDArray map(NDArray rhs, DoubleBinaryOperator operator) {
      DenseMatrix out = Cast.as(zeroLike());
      for (int i = 0; i < shape.matrixLength; i++) {
         out.matrix.data[i] = operator.applyAsDouble(matrix.data[i], rhs.get(i));
      }
      return out;
   }

   @Override
   public NDArray mapi(NDArray rhs, DoubleBinaryOperator operator) {
      for (int i = 0; i < shape.matrixLength; i++) {
         matrix.data[i] = operator.applyAsDouble(matrix.data[i], rhs.get(i));
      }
      return this;
   }

   @Override
   public NDArray mul(NDArray rhs) {
      checkLength(shape, rhs.shape());
      if (rhs.isDense()) {
         return new DenseMatrix(matrix.mul(rhs.toDoubleMatrix()[0]));
      }
      return super.mul(rhs);
   }

   @Override
   public NDArray muli(NDArray rhs) {
      checkLength(shape, rhs.shape());
      if (rhs.isDense()) {
         matrix.muli(rhs.toDoubleMatrix()[0]);
         return this;
      }
      return super.muli(rhs);
   }

   @Override
   public NDArray muliColumnVector(NDArray rhs) {
      if (rhs.isDense()) {
         matrix.muliColumnVector(rhs.toDoubleMatrix()[0]);
         return this;
      }
      return super.muliColumnVector(rhs);
   }

   @Override
   public NDArray muliRowVector(NDArray rhs) {
      if (rhs.isDense()) {
         matrix.muliRowVector(rhs.toDoubleMatrix()[0]);
         return this;
      }
      return super.muliRowVector(rhs);
   }

   @Override
   public NDArray rdiv(NDArray rhs) {
      checkLength(shape, rhs.shape());
      if (rhs.isDense()) {
         return new DenseMatrix(matrix.rdiv(rhs.toDoubleMatrix()[0]));
      }
      return super.rdiv(rhs);
   }

   @Override
   public NDArray rdivi(NDArray rhs) {
      checkLength(shape, rhs.shape());
      if (rhs.isDense()) {
         matrix.rdivi(rhs.toDoubleMatrix()[0]);
         return this;
      }
      return super.rdivi(rhs);
   }

   @Override
   public NDArray rsub(NDArray rhs) {
      checkLength(shape, rhs.shape());
      if (rhs.isDense()) {
         return new DenseMatrix(matrix.rsub(rhs.toDoubleMatrix()[0]));
      }
      return super.rsub(rhs);
   }

   @Override
   public NDArray rsubi(NDArray rhs) {
      checkLength(shape, rhs.shape());
      if (rhs.isDense()) {
         matrix.rsubi(rhs.toDoubleMatrix()[0]);
         return this;
      }
      return super.rsubi(rhs);
   }

   @Override
   public NDArray map(DoubleUnaryOperator operator) {
      DenseMatrix dm = new DenseMatrix(matrix.rows, matrix.columns);
      for (int i = 0; i < matrix.length; i++) {
         dm.matrix.data[i] = operator.applyAsDouble(matrix.data[i]);
      }
      return dm;
   }

   @Override
   public NDArray mapi(DoubleUnaryOperator operator) {
      for (int i = 0; i < matrix.length; i++) {
         matrix.data[i] = operator.applyAsDouble(matrix.data[i]);
      }
      return this;
   }

   @Override
   public NDArray trimToSize() {
      return this;
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
   public NDArray sub(NDArray rhs) {
      checkLength(shape, rhs.shape());
      if (rhs.isDense()) {
         return new DenseMatrix(matrix.sub(rhs.toDoubleMatrix()[0]));
      }
      return super.sub(rhs);
   }

   @Override
   public NDArray subi(NDArray rhs) {
      checkLength(shape, rhs.shape());
      if (rhs.isDense()) {
         matrix.subi(rhs.toDoubleMatrix()[0]);
         return this;
      }
      return super.subi(rhs);
   }

   @Override
   public NDArray subiColumnVector(NDArray rhs) {
      if (rhs.isDense()) {
         matrix.subiColumnVector(rhs.toDoubleMatrix()[0]);
         return this;
      }
      return super.subiColumnVector(rhs);
   }

   @Override
   public NDArray subiRowVector(NDArray rhs) {
      if (rhs.isDense()) {
         matrix.subiRowVector(rhs.toDoubleMatrix()[0]);
         return this;
      }
      return super.subiRowVector(rhs);
   }

   @Override
   public DoubleMatrix[] toDoubleMatrix() {
      return new DoubleMatrix[]{matrix};
   }

   @Override
   public NDArray zeroLike() {
      return new DenseMatrix(shape);
   }

   @Override
   public double sum() {
      return matrix.sum();
   }

   @Override
   public NDArray rowSums() {
      return new DenseMatrix(matrix.rowSums());
   }

   @Override
   public NDArray columnSums() {
      return new DenseMatrix(matrix.columnSums());
   }

   @Override
   public double min() {
      return matrix.min();
   }

   @Override
   public double max() {
      return matrix.max();
   }

   @Override
   public double argmin() {
      return matrix.argmin();
   }

   @Override
   public double argmax() {
      if( matrix.argmax() < 0 ){
         System.out.println(matrix);
      }
      return matrix.argmax();
   }

   @Override
   public NDArray getRow(int row) {
      return new DenseMatrix(matrix.getRow(row));
   }

   @Override
   public NDArray getColumn(int column) {
      return new DenseMatrix(matrix.getColumn(column));
   }

   @Override
   public NDArray setColumn(int column, NDArray array) {
      checkLength(shape.rows(), array.shape());
      matrix.putColumn(column, array.toDoubleMatrix()[0]);
      return this;
   }

   @Override
   public NDArray setRow(int row, NDArray array) {
      checkLength(shape.columns(), array.shape());
      matrix.putRow(row, array.toDoubleMatrix()[0]);
      return this;
   }

   @Override
   public NDArray reshape(int... dims) {
      shape.reshape(dims);
      matrix.reshape(shape.rows(), shape.columns());
      return this;
   }
}//END OF DenseTwoDArray
