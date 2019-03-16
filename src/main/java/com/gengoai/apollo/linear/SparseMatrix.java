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

import com.gengoai.Copyable;
import com.gengoai.concurrent.AtomicDouble;
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

   @Override
   public NDArray getSubMatrix(int fromRow, int toRow, int fromCol, int toCol) {
      SparseMatrix sm = new SparseMatrix((toRow - fromRow), (toCol - fromCol));
      map.forEachPair((i, v) -> {
         int row = i % shape.rows();
         int col = i / shape.rows();
         if (row >= fromRow && row < toRow
                && col >= fromCol && col < toCol) {
            sm.set(row - fromRow, col - fromCol, v);
         }
         return true;
      });
      return sm;
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
            map.adjustOrPutValue(i, -v, -v);
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
   public NDArray zero() {
      map.clear();
      map.trimToSize();
      return this;
   }

   @Override
   public NDArray set(long i, double value) {
      if (value == 0) {
         map.removeKey((int) i);
      } else {
         map.put((int) i, value);
      }
      return this;
   }

   @Override
   public NDArray set(int row, int col, double value) {
      map.put(index(row, col), value);
      return this;
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
   public NDArray compact() {
      map.trimToSize();
      return this;
   }

   @Override
   public NDArray zeroLike() {
      return new SparseMatrix(shape);
   }

   @Override
   public double[] toDoubleArray() {
      double[] array = new double[(int) length()];
      map.forEachPair((i, v) -> {
         array[i] = v;
         return true;
      });
      return array;
   }

   @Override
   public void forEachSparse(EntryConsumer consumer) {
      map.forEachPair((i, v) -> {
         consumer.apply(i, v);
         return true;
      });
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
   public double norm1() {
      double sum = 0;
      for (double element : map.values().elements()) {
         sum += Math.abs(element);
      }
      return sum;
   }

   @Override
   public double sumOfSquares() {
      double sum = 0;
      for (double element : map.values().elements()) {
         sum += element * element;
      }
      return sum;
   }

   @Override
   public double dot(NDArray rhs) {
      checkLength(shape, rhs.shape());
      final AtomicDouble dot = new AtomicDouble(0d);
      map.forEachPair((i, v) -> {
         dot.addAndGet(rhs.get(i) * v);
         return true;
      });
      return dot.get();
   }

   @Override
   public long size() {
      return map.size();
   }

   @Override
   public NDArray getRow(int row) {
      SparseMatrix sm = new SparseMatrix(1, shape.columns());
      for (int i = 0; i < shape.columns(); i++) {
         sm.set(row, i, get(row, i));
      }
      return sm;
   }

   @Override
   public NDArray getColumn(int column) {
      SparseMatrix sm = new SparseMatrix(shape.rows(), 1);
      for (int i = 0; i < shape.rows(); i++) {
         sm.set(i, get(i, column));
      }
      return sm;
   }

   @Override
   public NDArray setColumn(int column, NDArray array) {
      checkLength(shape.rows(), array.shape());
      for (int i = 0; i < array.shape().matrixLength; i++) {
         set(i, column, array.get(i));
      }
      return this;
   }

   @Override
   public NDArray setRow(int row, NDArray array) {
      checkLength(shape.columns(), array.shape());
      for (int i = 0; i < array.shape().matrixLength; i++) {
         set(row, i, array.get(i));
      }
      return this;
   }

   @Override
   public NDArray reshape(int... dims) {
      shape.reshape(dims);
      return this;
   }
}//END OF SparseMatrix
