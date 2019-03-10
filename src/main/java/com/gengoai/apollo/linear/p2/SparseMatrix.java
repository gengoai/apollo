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

import com.gengoai.Copyable;
import com.gengoai.apollo.linear.Shape;
import com.gengoai.conversion.Cast;
import com.gengoai.math.Math2;
import com.gengoai.math.Optimum;
import org.apache.mahout.math.map.OpenIntDoubleHashMap;
import org.jblas.DoubleMatrix;

import java.util.function.DoubleUnaryOperator;

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
      this.map = Copyable.deepCopy(toCopy.map);
   }


   public static SparseMatrix rand(int rows, int cols) {
      return rand(rows, cols, 0.95);
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

   @Override
   public NDArray map(DoubleUnaryOperator operator) {
      NDArray out = zeroLike();
      for (int i = 0; i < shape.matrixLength; i++) {
         out.set(i, operator.applyAsDouble(get(i)));
      }
      return out;
   }

   @Override
   public NDArray mapi(DoubleUnaryOperator operator) {
      for (int i = 0; i < shape.matrixLength; i++) {
         set(i, operator.applyAsDouble(get(i)));
      }
      return this;
   }

   @Override
   public NDArray copy() {
      SparseMatrix sm = new SparseMatrix(this);
      return sm;
   }

   @Override
   public NDArray add(NDArray rhs) {
      return copy().addi(rhs);
   }

   @Override
   public NDArray div(NDArray rhs) {
      return copy().divi(rhs);
   }

   @Override
   public NDArray mul(NDArray rhs) {
      return copy().muli(rhs);
   }

   @Override
   public NDArray sub(NDArray rhs) {
      return copy().subi(rhs);
   }

   @Override
   public NDArray addi(NDArray rhs) {
      if (!rhs.isDense()) {
         checkLength(shape, rhs.shape());
         SparseMatrix sm = Cast.as(rhs);
         sm.map.forEachPair((i, v) -> {
            map.adjustOrPutValue(i, v, v);
            return true;
         });
         return this;
      }
      return super.addi(rhs);
   }

   @Override
   public NDArray subi(NDArray rhs) {
      if (!rhs.isDense()) {
         checkLength(shape, rhs.shape());
         SparseMatrix sm = Cast.as(rhs);
         sm.map.forEachPair((i, v) -> {
            map.adjustOrPutValue(i, v, -v);
            return true;
         });
         return this;
      }
      return super.addi(rhs);
   }

   @Override
   public NDArray muli(NDArray rhs) {
      checkLength(shape, rhs.shape());
      map.forEachPair((i, v) -> {
         map.put(i, v * rhs.get(i));
         return true;
      });
      return this;
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
      if (value == 0) {
         map.removeKey((int) i);
      } else {
         map.put((int) i, value);
      }
   }

   @Override
   public void set(int row, int col, double value) {
      map.put(index(row, col), value);
   }

   @Override
   public DoubleMatrix[] toDoubleMatrix() {
      DoubleMatrix m = new DoubleMatrix(shape.rows(), shape.columns());
      map.forEachPair((i, v) -> {
         m.data[i] = v;
         return true;
      });
      return new DoubleMatrix[]{m};
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
         t = new SparseMatrix(shape.columns(), shape.rows());
         map.forEachPair((i, v) -> {
            int row = i % shape.rows();
            int col = i / shape.rows();
            t.set(col, row, v);
            return true;
         });
      }
      return t;
   }

   @Override
   public double sum() {
      return Math2.sum(map.values().elements());
   }

   @Override
   public double min() {
      double min = Optimum.MINIMUM.optimumValue(map.values().elements());
      if (map.size() == shape.matrixLength) {
         return min;
      }
      return Math.min(0, min);
   }

   @Override
   public double max() {
      double max = Optimum.MAXIMUM.optimumValue(map.values().elements());
      if (map.size() == shape.matrixLength) {
         return max;
      }
      return Math.max(0, max);
   }


   @Override
   public long size() {
      return map.size();
   }
}//END OF SparseMatrix
