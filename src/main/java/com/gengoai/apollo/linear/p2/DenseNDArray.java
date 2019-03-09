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
import com.gengoai.conversion.Cast;
import org.jblas.FloatMatrix;

/**
 * @author David B. Bracewell
 */
public class DenseNDArray extends NDArray {
   private final FloatMatrix matrix;

   protected DenseNDArray(int... dims) {
      this(new Shape(dims));
   }

   protected DenseNDArray(Shape shape) {
      super(shape);
      this.matrix = new FloatMatrix(shape.rows(), shape.columns());
   }

   protected DenseNDArray(FloatMatrix fm) {
      super(new Shape(fm.rows, fm.columns));
      this.matrix = fm;
   }

   @Override
   public NDArray zeroLike() {
      return new DenseNDArray(shape);
   }

   @Override
   public NDArray addRowVector(NDArray rhs) {
      if (rhs.isDense()) {
         return new DenseNDArray(matrix.addRowVector(rhs.asDense().matrix));
      }
      return null;
   }

   @Override
   public NDArray addColumnVector(NDArray rhs) {
      if (rhs.isDense()) {
         return new DenseNDArray(matrix.addColumnVector(rhs.asDense().matrix));
      }
      return null;
   }

   @Override
   public NDArray addiRowVector(NDArray rhs) {
      if (rhs.isDense()) {
         matrix.addiRowVector(rhs.asDense().matrix);
         return this;
      }
      return null;
   }

   @Override
   public NDArray addiColumnVector(NDArray rhs) {
      if (rhs.isDense()) {
         matrix.addiColumnVector(rhs.asDense().matrix);
         return this;
      }
      return null;
   }

   @Override
   public double get(int i) {
      return matrix.get(i);
   }

   @Override
   public double get(int row, int col) {
      return matrix.get(row, col);
   }

   @Override
   public void set(int i, double value) {
      matrix.put(i, (float) value);
   }

   @Override
   public void set(int row, int col, double value) {
      matrix.put(row, col, (float) value);
   }

   @Override
   protected void forEach(EntryConsumer consumer) {
      for (int i = 0; i < matrix.length; i++) {
         consumer.apply(i, matrix.data[i]);
      }
   }

   @Override
   protected void forEachSparse(EntryConsumer consumer) {
      for (int i = 0; i < matrix.length; i++) {
         if (matrix.data[i] != 0) {
            consumer.apply(i, matrix.data[i]);
         }
      }
   }

   public static DenseNDArray rand(int rows, int columns) {
      return new DenseNDArray(FloatMatrix.rand(rows, columns));
   }

   @Override
   protected NDArray addi(NDArray rhs, NDArray out) {
      if (rhs.isDense()) {
         DenseNDArray dr = Cast.as(rhs);
         DenseNDArray dt = Cast.as(out);
         matrix.addi(dr.matrix, dt.matrix);
         return dt;
      }
      return super.addi(rhs, out);
   }

   public DenseNDArray asDense() {
      return this;
   }

   @Override
   public boolean isDense() {
      return true;
   }

   public void adjust(int i, double incrementBy) {
      matrix.data[i] += incrementBy;
   }

   public static void loop() {
      DenseNDArray n1 = rand(10, 10);
      DenseNDArray n2 = rand(1, 10);
      for (int i = 0; i < 10_000; i++) {
         for (int i1 = 0; i1 < n1.matrix.length; i1++) {
            n1.matrix.data[i1]++;
         }
//         for (int i1 = 0; i1 < n1.matrix.length; i1++) {
//            n1.set(i1, n1.get(i1) + 1);
//         }
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

}//END OF DenseNDArray
