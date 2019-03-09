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

package com.gengoai.apollo.linear.p2;

import com.gengoai.Stopwatch;
import com.gengoai.apollo.linear.Shape;
import com.gengoai.concurrent.Threads;
import org.jblas.DoubleMatrix;

/**
 * @author David B. Bracewell
 */
public class DenseMatrix extends Matrix {
   private final DoubleMatrix matrix;

   protected DenseMatrix(int... dims) {
      this(new Shape(dims));
   }

   protected DenseMatrix(Shape shape) {
      super(shape);
      this.matrix = new DoubleMatrix(shape.rows(), shape.columns());
   }

   protected DenseMatrix(DoubleMatrix fm) {
      super(new Shape(fm.rows, fm.columns));
      this.matrix = fm;
   }

   public static void loop() {
      DenseMatrix n1 = rand(1000, 1000);
      DenseMatrix n4 = rand(1000, 1000);
      DenseMatrix n2 = rand(1, 1000);
      DenseMatrix n3 = rand(1000, 1);
      for (int i = 0; i < 10_000; i++) {
//         n1.addiColumnVector(n3);
//         n1.addiRowVector(n2);
         n1.addi(n4);
      }
   }

   public static void main(String[] args) throws Exception {
      loop();
      System.gc();
      Threads.sleep(10_000);
      Stopwatch sw = Stopwatch.createStarted();
      loop();
      System.out.println(sw);
   }

   public static DenseMatrix rand(int rows, int columns) {
      return new DenseMatrix(DoubleMatrix.rand(rows, columns));
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
      checkLength(shape, rhs.shape());
      if (rhs.isDense()) {
         return new DenseMatrix(matrix.div(rhs.toDoubleMatrix()[0]));
      }
      return super.add(rhs);
   }

   @Override
   public NDArray divi(NDArray rhs) {
      checkLength(shape, rhs.shape());
      if (rhs.isDense()) {
         matrix.divi(rhs.toDoubleMatrix()[0]);
         return this;
      }
      return super.add(rhs);
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
   public void forEach(EntryConsumer consumer) {
      for (int i = 0; i < matrix.length; i++) {
         consumer.apply(i, matrix.data[i]);
      }
   }

   @Override
   public void forEachSparse(EntryConsumer consumer) {
      for (int i = 0; i < matrix.length; i++) {
         if (matrix.data[i] != 0) {
            consumer.apply(i, matrix.data[i]);
         }
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
   public boolean isDense() {
      return true;
   }

   @Override
   public NDArray mul(NDArray rhs) {
      checkLength(shape, rhs.shape());
      if (rhs.isDense()) {
         return new DenseMatrix(matrix.mul(rhs.toDoubleMatrix()[0]));
      }
      return super.add(rhs);
   }

   @Override
   public NDArray muli(NDArray rhs) {
      checkLength(shape, rhs.shape());
      if (rhs.isDense()) {
         matrix.muli(rhs.toDoubleMatrix()[0]);
         return this;
      }
      return super.add(rhs);
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
      return super.add(rhs);
   }

   @Override
   public NDArray rdivi(NDArray rhs) {
      checkLength(shape, rhs.shape());
      if (rhs.isDense()) {
         matrix.rdivi(rhs.toDoubleMatrix()[0]);
         return this;
      }
      return super.add(rhs);
   }

   @Override
   public NDArray rsub(NDArray rhs) {
      checkLength(shape, rhs.shape());
      if (rhs.isDense()) {
         return new DenseMatrix(matrix.rsub(rhs.toDoubleMatrix()[0]));
      }
      return super.add(rhs);
   }

   @Override
   public NDArray rsubi(NDArray rhs) {
      checkLength(shape, rhs.shape());
      if (rhs.isDense()) {
         matrix.rsubi(rhs.toDoubleMatrix()[0]);
         return this;
      }
      return super.add(rhs);
   }

   @Override
   public void set(long i, double value) {
      matrix.put((int) i, value);
   }

   @Override
   public void set(int row, int col, double value) {
      matrix.put(row, col, value);
   }

   @Override
   public NDArray sub(NDArray rhs) {
      checkLength(shape, rhs.shape());
      if (rhs.isDense()) {
         return new DenseMatrix(matrix.sub(rhs.toDoubleMatrix()[0]));
      }
      return super.add(rhs);
   }

   @Override
   public NDArray subi(NDArray rhs) {
      checkLength(shape, rhs.shape());
      if (rhs.isDense()) {
         matrix.subi(rhs.toDoubleMatrix()[0]);
         return this;
      }
      return super.add(rhs);
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

}//END OF DenseTwoDArray
