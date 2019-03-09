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
import com.gengoai.Validation;
import com.gengoai.apollo.linear.Shape;
import com.gengoai.concurrent.Threads;
import com.gengoai.conversion.Cast;
import org.jblas.FloatMatrix;

/**
 * @author David B. Bracewell
 */
public class DenseTwoDArray extends TwoDArray {
   private final FloatMatrix matrix;

   protected DenseTwoDArray(int... dims) {
      this(new Shape(dims));
   }

   protected DenseTwoDArray(Shape shape) {
      super(shape);
      this.matrix = new FloatMatrix(shape.rows(), shape.columns());
   }

   protected DenseTwoDArray(FloatMatrix fm) {
      super(new Shape(fm.rows, fm.columns));
      this.matrix = fm;
   }

   @Override
   public NDArray zeroLike() {
      return new DenseTwoDArray(shape);
   }

   @Override
   public NDArray mul(NDArray rhs) {
      return null;
   }

   @Override
   public NDArray mulColumnVector(NDArray rhs) {
      return null;
   }

   @Override
   public NDArray mulRowVector(NDArray rhs) {
      return null;
   }

   @Override
   public NDArray muli(NDArray rhs) {
      return null;
   }

   @Override
   public NDArray muliColumnVector(NDArray rhs) {
      return null;
   }

   @Override
   public NDArray muliRowVector(NDArray rhs) {
      return null;
   }

   @Override
   public NDArray div(NDArray rhs) {
      return null;
   }

   @Override
   public NDArray divColumnVector(NDArray rhs) {
      return null;
   }

   @Override
   public NDArray divRowVector(NDArray rhs) {
      return null;
   }

   @Override
   public NDArray divi(NDArray rhs) {
      return null;
   }

   @Override
   public NDArray diviColumnVector(NDArray rhs) {
      return null;
   }

   @Override
   public NDArray diviRowVector(NDArray rhs) {
      return null;
   }

   @Override
   public NDArray rdiv(NDArray rhs) {
      return null;
   }

   @Override
   public NDArray rdivColumnVector(NDArray rhs) {
      return null;
   }

   @Override
   public NDArray rdivRowVector(NDArray rhs) {
      return null;
   }

   @Override
   public NDArray rdivi(NDArray rhs) {
      return null;
   }

   @Override
   public NDArray rdiviColumnVector(NDArray rhs) {
      return null;
   }

   @Override
   public NDArray rdiviRowVector(NDArray rhs) {
      return null;
   }

   @Override
   public NDArray addRowVector(NDArray rhs) {
      Validation.checkArgument(rhs instanceof TwoDArray, () -> sizeMismatch(rhs.shape()));
      if (rhs.isDense()) {
         DenseTwoDArray dense = Cast.as(rhs);
         return new DenseTwoDArray(matrix.addRowVector(dense.matrix));
      }
      return null;
   }

   @Override
   public NDArray addColumnVector(NDArray rhs) {
      if (rhs.isDense()) {
         DenseTwoDArray dense = Cast.as(rhs);
         return new DenseTwoDArray(matrix.addColumnVector(dense.matrix));
      }
      return null;
   }

   @Override
   public NDArray addiRowVector(NDArray rhs) {
      if (rhs.isDense()) {
         DenseTwoDArray dense = Cast.as(rhs);
         matrix.addiRowVector(dense.matrix);
         return this;
      }
      return null;
   }

   @Override
   public NDArray rsub(NDArray rhs) {
      return null;
   }

   @Override
   public NDArray rsubColumnVector(NDArray rhs) {
      return null;
   }

   @Override
   public NDArray rsubRowVector(NDArray rhs) {
      return null;
   }

   @Override
   public NDArray rsubi(NDArray rhs) {
      return null;
   }

   @Override
   public NDArray rsubiColumnVector(NDArray rhs) {
      return null;
   }

   @Override
   public NDArray rsubiRowVector(NDArray rhs) {
      return null;
   }

   @Override
   public NDArray slice(int slice) {
      return null;
   }

   @Override
   public NDArray sub(NDArray rhs) {
      return null;
   }

   @Override
   public NDArray subColumnVector(NDArray rhs) {
      return null;
   }

   @Override
   public NDArray subRowVector(NDArray rhs) {
      return null;
   }

   @Override
   public NDArray subi(NDArray rhs) {
      return null;
   }

   @Override
   public NDArray subiColumnVector(NDArray rhs) {
      return null;
   }

   @Override
   public NDArray subiRowVector(NDArray rhs) {
      return null;
   }

   @Override
   public NDArray addiColumnVector(NDArray rhs) {
      if (rhs.isDense()) {
         DenseTwoDArray dense = Cast.as(rhs);
         matrix.addiColumnVector(dense.matrix);
         return this;
      }
      return null;
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
   public void set(long i, double value) {
      matrix.put((int) i, (float) value);
   }

   @Override
   public void set(int row, int col, double value) {
      matrix.put(row, col, (float) value);
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

   public static DenseTwoDArray rand(int rows, int columns) {
      return new DenseTwoDArray(FloatMatrix.rand(rows, columns));
   }

   @Override
   protected NDArray addi(NDArray rhs, NDArray out) {
      if (rhs.isDense()) {
         DenseTwoDArray dr = Cast.as(rhs);
         DenseTwoDArray dt = Cast.as(out);
         matrix.addi(dr.matrix, dt.matrix);
         return dt;
      }
      return super.addi(rhs, out);
   }

   public DenseTwoDArray asDense() {
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
      DenseTwoDArray n1 = rand(10, 10);
      DenseTwoDArray n2 = rand(10, 1);
      for (int i = 0; i < 10_000; i++) {
         n1.add(n2);
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

}//END OF DenseTwoDArray
