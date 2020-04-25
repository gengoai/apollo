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

import com.gengoai.Copyable;
import com.gengoai.concurrent.AtomicDouble;
import com.gengoai.conversion.Cast;
import com.gengoai.math.Math2;
import com.gengoai.math.Optimum;
import org.apache.mahout.math.list.IntArrayList;
import org.apache.mahout.math.map.OpenIntDoubleHashMap;
import org.apache.mahout.math.set.OpenIntHashSet;
import org.jblas.DoubleMatrix;

import java.util.function.DoubleUnaryOperator;
import java.util.stream.IntStream;

/**
 * Sparse Matrix implementation using a single hashmap.
 *
 * @author David B. Bracewell
 */
public class SparseMatrix extends Matrix {
   private static final long serialVersionUID = 1L;
   private final OpenIntDoubleHashMap map;

   /**
    * Instantiates a new Sparse matrix.
    *
    * @param dims the dims
    */
   public SparseMatrix(int... dims) {
      this(new Shape(dims));
   }

   /**
    * Instantiates a new Sparse matrix.
    *
    * @param shape the shape
    */
   public SparseMatrix(Shape shape) {
      super(shape);
      this.map = new OpenIntDoubleHashMap();
   }


   /**
    * Instantiates a new Sparse matrix.
    *
    * @param toCopy the to copy
    */
   protected SparseMatrix(SparseMatrix toCopy) {
      super(toCopy.shape);
      this.map = Copyable.deepCopy(toCopy.map);
   }

   @Override
   public NDArray selectColumns(int... indices) {
      OpenIntHashSet adj = new OpenIntHashSet();
      IntStream.of(indices)
               .filter(i -> i >= 0 && i < columns())
               .forEach(adj::add);
      SparseMatrix out = new SparseMatrix(rows(), adj.size());
      map.forEachPair((i, d) -> {
         int c = shape.toColumn(i);
         if(adj.contains(c)) {
            out.set(i, d);
         }
         return true;
      });
      return out;
   }

   @Override
   public int[] sparseIndices() {
      IntArrayList ial = map.keys();
      ial.sort();
      return ial.toArray(new int[0]);
   }

   @Override
   public NDArray selectRows(int... indices) {
      OpenIntHashSet adj = new OpenIntHashSet();
      IntStream.of(indices)
               .filter(i -> i >= 0 && i < rows())
               .forEach(adj::add);
      SparseMatrix out = new SparseMatrix(adj.size(), columns());
      map.forEachPair((i, d) -> {
         int c = shape.toRow(i);
         if(adj.contains(c)) {
            out.set(i, d);
         }
         return true;
      });
      return out;
   }

   @Override
   public NDArray T() {
      SparseMatrix t;
      if(shape.isVector()) {
         t = new SparseMatrix(this);
         t.shape.reshape(shape.columns(), shape.rows());
      }
      else {
         t = new SparseMatrix(shape.columns(), shape.rows());
         map.forEachPair((i, v) -> {
            t.set(shape.toColumn(i), shape.toRow(i), v);
            return true;
         });
      }
      return t;
   }

   @Override
   public NDArray add(NDArray rhs) {
      return copy().addi(rhs);
   }

   @Override
   public NDArray addi(NDArray rhs) {
      if(!rhs.isDense()) {
         checkLength(rhs.shape);
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
   public NDArray compact() {
      map.trimToSize();
      return this;
   }

   @Override
   public NDArray div(NDArray rhs) {
      return copy().divi(rhs);
   }

   @Override
   public double dot(NDArray rhs) {
      checkLength(rhs.shape());
      final AtomicDouble dot = new AtomicDouble(0d);
      map.forEachPair((i, v) -> {
         dot.addAndGet(rhs.get(i) * v);
         return true;
      });
      return dot.get();
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

   @Override
   public double get(int row, int col) {
      return map.get(shape.matrixIndex(row, col));
   }

   @Override
   public NDArray getColumn(int column) {
      SparseMatrix sm = new SparseMatrix(shape.rows(), 1);
      for(int i = 0; i < shape.rows(); i++) {
         sm.set(i, get(i, column));
      }
      return sm;
   }

   @Override
   public NDArray getRow(int row) {
      SparseMatrix sm = new SparseMatrix(1, shape.columns());
      for(int i = 0; i < shape.columns(); i++) {
         sm.set(row, i, get(row, i));
      }
      return sm;
   }

   @Override
   public NDArray getSubMatrix(int fromRow, int toRow, int fromCol, int toCol) {
      SparseMatrix sm = new SparseMatrix((toRow - fromRow), (toCol - fromCol));
      map.forEachPair((i, v) -> {
         int row = shape.toRow(i);
         int col = shape.toColumn(i);
         if(row >= fromRow && row < toRow
               && col >= fromCol && col < toCol) {
            sm.set(row - fromRow, col - fromCol, v);
         }
         return true;
      });
      return sm;
   }

   @Override
   public boolean isDense() {
      return false;
   }

   @Override
   public NDArray map(DoubleUnaryOperator operator) {
      NDArray out = zeroLike();
      for(int i = 0; i < shape.matrixLength; i++) {
         out.set(i, operator.applyAsDouble(get(i)));
      }
      return out;
   }

   @Override
   public NDArray mapi(DoubleUnaryOperator operator) {
      for(int i = 0; i < shape.matrixLength; i++) {
         set(i, operator.applyAsDouble(get(i)));
      }
      return this;
   }

   @Override
   public double max() {
      double max = Optimum.MAXIMUM.optimumValue(map.values().elements());
      if(map.size() == shape.matrixLength) {
         return max;
      }
      return Math.max(0, max);
   }

   @Override
   public double min() {
      double min = Optimum.MINIMUM.optimumValue(map.values().elements());
      if(map.size() == shape.matrixLength) {
         return min;
      }
      return Math.min(0, min);
   }

   @Override
   public NDArray mul(NDArray rhs) {
      return copy().muli(rhs);
   }

   @Override
   public NDArray muli(NDArray rhs) {
      checkLength(rhs.shape());
      map.forEachPair((i, v) -> {
         map.put(i, v * rhs.get(i));
         return true;
      });
      return this;
   }

   @Override
   public double norm1() {
      double sum = 0;
      for(double element : map.values().elements()) {
         sum += Math.abs(element);
      }
      return sum;
   }

   @Override
   public NDArray reshape(int... dims) {
      shape.reshape(dims);
      return this;
   }

   @Override
   public NDArray set(long i, double value) {
      if(value == 0) {
         map.removeKey((int) i);
      }
      else {
         map.put((int) i, value);
      }
      return this;
   }

   @Override
   public NDArray set(int row, int col, double value) {
      map.put(shape.matrixIndex(row, col), value);
      return this;
   }

   @Override
   public NDArray setColumn(int column, NDArray array) {
      checkLength(shape.rows(), array.shape());
      for(int i = 0; i < array.shape().matrixLength; i++) {
         set(i, column, array.get(i));
      }
      return this;
   }

   @Override
   public NDArray setRow(int row, NDArray array) {
      checkLength(shape.columns(), array.shape());
      for(int i = 0; i < array.shape().matrixLength; i++) {
         set(row, i, array.get(i));
      }
      return this;
   }

   @Override
   public long size() {
      return map.size();
   }

   @Override
   public NDArray sub(NDArray rhs) {
      return copy().subi(rhs);
   }

   @Override
   public NDArray subi(NDArray rhs) {
      if(!rhs.isDense()) {
         checkLength(rhs.shape());
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
   public double sum() {
      return Math2.sum(map.values().elements());
   }

   @Override
   public double sumOfSquares() {
      double sum = 0;
      for(double element : map.values().elements()) {
         sum += element * element;
      }
      return sum;
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
   public DoubleMatrix[] toDoubleMatrix() {
      DoubleMatrix m = new DoubleMatrix(shape.rows(), shape.columns());
      map.forEachPair((i, v) -> {
         m.data[i] = v;
         return true;
      });
      return new DoubleMatrix[]{m};
   }

   @Override
   public NDArray zero() {
      map.clear();
      map.trimToSize();
      return this;
   }

   @Override
   public NDArray zeroLike() {
      return new SparseMatrix(shape);
   }


   @Override
   public NDArray mmul(NDArray rhs) {
      if(rhs.isDense() || (sparsity() < 0.5 && length() > 10_000)) {
         return super.mmul(rhs);
      }
      SparseMatrix product = new SparseMatrix(rows(), rhs.columns());
      map.forEachPair((i, v) -> {
         int row = shape.toRow(i);
         int colLHS = shape.toColumn(i);
         for(int colRHS = 0; colRHS < rhs.columns(); colRHS++) {
            double prod = v * rhs.get(colLHS, colRHS);
            product.map.adjustOrPutValue(product.shape.matrixIndex(row, colRHS), prod, prod);
         }
         return true;
      });
      return product;
   }
}//END OF SparseMatrix