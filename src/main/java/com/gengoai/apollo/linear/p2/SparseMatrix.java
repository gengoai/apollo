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
import com.gengoai.apollo.linear.NDArrayFactory;
import com.gengoai.apollo.linear.NDArrayInitializer;
import com.gengoai.apollo.linear.Shape;
import com.gengoai.concurrent.Threads;
import org.apache.mahout.math.map.OpenIntDoubleHashMap;
import org.jblas.DoubleMatrix;

/**
 * @author David B. Bracewell
 */
public class SparseMatrix extends Matrix {
   private final OpenIntDoubleHashMap map;

   public SparseMatrix(int... dims) {
      this(new Shape(dims));
   }

   public SparseMatrix(Shape shape) {
      super(shape);
      this.map = new OpenIntDoubleHashMap();
   }

   protected SparseMatrix(SparseMatrix toCopy) {
      super(toCopy.shape);
      this.map = new OpenIntDoubleHashMap();
      toCopy.map.forEachPair((i, v) -> {
         this.map.put(i, v);
         return true;
      });
   }


   public static SparseMatrix rand(int rows, int cols) {
      return rand(rows, cols, 0.5);
   }

   public static SparseMatrix rand(int rows, int cols, double sparsity) {
      SparseMatrix matrix = new SparseMatrix(rows, cols);
      for (int i = 0; i < matrix.shape.matrixLength; i++) {
         if (Math.random() > sparsity) {
            matrix.set(i, Math.random());
         }
      }
      return matrix;
   }

   public static void loop() {
      SparseMatrix n1 = rand(100, 100);
      SparseMatrix n4 = rand(100, 100);
      SparseMatrix n2 = rand(1, 1000);
      SparseMatrix n3 = rand(1000, 1);
      for (int i = 0; i < 10_000; i++) {
//         n1.addiColumnVector(n3);
//         n1.addiRowVector(n2);
         n1.add(n4);
      }
   }

   public static void loop2() {
      com.gengoai.apollo.linear.NDArray n1 = NDArrayFactory.SPARSE.create(NDArrayInitializer.rand, 1000, 1000);
      com.gengoai.apollo.linear.NDArray n4 = NDArrayFactory.SPARSE.create(NDArrayInitializer.rand, 1000, 1000);
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

   @Override
   public NDArray copy() {
      SparseMatrix sm = new SparseMatrix(this);
      return sm;
   }


   @Override
   public void forEachSparse(EntryConsumer consumer) {
      map.forEachPair((i, v) -> {
         consumer.apply(i, v);
         return true;
      });
   }

   @Override
   public double get(long i) {
      return map.get((int) i);
   }

   public int index(int rowIndex, int columnIndex) {
      return rowIndex + shape.rows() * columnIndex;
   }

   @Override
   public double get(int row, int col) {
      return map.get(index(row, col));
   }

   @Override
   public boolean isDense() {
      return false;
   }

   @Override
   public void set(long i, double value) {
      map.put((int) i, value);
   }

   @Override
   public void set(int row, int col, double value) {
      map.put(index(row, col), value);
   }

   @Override
   public DoubleMatrix[] toDoubleMatrix() {
      return new DoubleMatrix[0];
   }

   @Override
   public NDArray zeroLike() {
      return new SparseMatrix();
   }

   @Override
   public NDArray T() {
      SparseMatrix t;
      if (shape.isVector()) {
         t = new SparseMatrix(this);
         t.shape.reshape(shape.columns(), shape.rows());
      } else {
         t = new SparseMatrix();
         forEachSparse((i, v) -> {
            int row = (int) i % shape.rows();
            int col = (int) i / shape.rows();
            t.set(col, row, v);
         });
      }
      return t;
   }
}//END OF SparseMatrix
