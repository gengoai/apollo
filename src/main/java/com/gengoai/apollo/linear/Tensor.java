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

import org.jblas.DoubleMatrix;

import java.util.function.DoubleBinaryOperator;
import java.util.function.DoublePredicate;
import java.util.function.DoubleUnaryOperator;
import java.util.stream.Stream;

/**
 * @author David B. Bracewell
 */
public class Tensor extends NDArray {
   private static final long serialVersionUID = 1L;
   final NDArray[] slices;

   public Tensor(int kernels, int channels, NDArray[] slices) {
      super(Shape.shape(kernels, channels, slices[0].rows(), slices[0].columns()));
      this.slices = slices;
   }

   private Tensor(Shape shape) {
      super(shape);
      this.slices = new NDArray[shape.sliceLength];
   }

   @Override
   public NDArray T() {
      Tensor tensor = new Tensor(Shape.shape(kernels(), channels(), columns(), rows()));
      for (int i = 0; i < slices.length; i++) {
         tensor.slices[i] = slices[i].T();
      }
      return tensor;
   }


   private void check(Shape shape) {
      if (shape.sliceLength > 1 && shape.sliceLength != shape().sliceLength) {
         throw new IllegalArgumentException(
            "Invalid Slice Length: " + shape.sliceLength + " != " + shape().sliceLength);
      }
      if (shape.matrixLength != shape().matrixLength) {
         throw new IllegalArgumentException(
            "Invalid Matrix Length: " + shape.matrixLength + " != " + shape().matrixLength);
      }
   }

   private void check(int target, Shape shape) {
      if (shape.sliceLength > 1 && shape.sliceLength != shape().sliceLength) {
         throw new IllegalArgumentException(
            "Invalid Slice Length: " + shape.sliceLength + " != " + shape().sliceLength);
      }
      if (shape.matrixLength != target) {
         throw new IllegalArgumentException(
            "Invalid Matrix Length: " + shape.matrixLength + " != " + target);
      }
   }

   @Override
   public NDArray add(double value) {
      Tensor tensor = new Tensor(shape);
      for (int i = 0; i < slices.length; i++) {
         tensor.slices[i] = slices[i].add(value);
      }
      return tensor;
   }

   @Override
   public NDArray add(NDArray rhs) {
      check(rhs.shape);
      Tensor tensor = new Tensor(shape);
      for (int i = 0; i < slices.length; i++) {
         tensor.slices[i] = slices[i].add(rhs.slice(i));
      }
      return tensor;
   }

   @Override
   public NDArray addColumnVector(NDArray rhs) {
      check(shape.rows(), rhs.shape);
      Tensor tensor = new Tensor(shape);
      for (int i = 0; i < slices.length; i++) {
         tensor.slices[i] = slices[i].addColumnVector(rhs.slice(i));
      }
      return tensor;
   }

   @Override
   public NDArray addRowVector(NDArray rhs) {
      check(shape.columns(), rhs.shape);
      Tensor tensor = new Tensor(shape);
      for (int i = 0; i < slices.length; i++) {
         tensor.slices[i] = slices[i].addRowVector(rhs.slice(i));
      }
      return tensor;
   }

   @Override
   public NDArray addi(double value) {
      for (NDArray slice : slices) {
         slice.addi(value);
      }
      return this;
   }

   @Override
   public NDArray addi(NDArray rhs) {
      check(rhs.shape);
      for (int i = 0; i < slices.length; i++) {
         slices[i].addi(rhs.slice(i));
      }
      return this;
   }

   @Override
   public NDArray addiColumnVector(NDArray rhs) {
      check(rows(), rhs.shape);
      for (int i = 0; i < slices.length; i++) {
         slices[i].addiColumnVector(rhs.slice(i));
      }
      return this;
   }

   @Override
   public NDArray addiRowVector(NDArray rhs) {
      check(columns(), rhs.shape);
      for (int i = 0; i < slices.length; i++) {
         slices[i].addiRowVector(rhs.slice(i));
      }
      return this;
   }

   @Override
   public long argmax() {
      long index = 0;
      double max = Double.NEGATIVE_INFINITY;
      for (int i = 0; i < slices.length; i++) {
         long im = slices[i].argmax();
         double v = slices[i].get(im);
         if (v > max) {
            index = im * i;
            max = v;
         }
      }
      return index;
   }

   @Override
   public long argmin() {
      long index = 0;
      double min = Double.POSITIVE_INFINITY;
      for (int i = 0; i < slices.length; i++) {
         long im = slices[i].argmin();
         double v = slices[i].get(im);
         if (v < min) {
            index = im * i;
            min = v;
         }
      }
      return index;
   }

   @Override
   public NDArray columnArgmaxs() {
      Tensor tensor = new Tensor(Shape.shape(kernels(), channels(), 1, columns()));
      for (int i = 0; i < slices.length; i++) {
         tensor.slices[i] = slices[i].columnArgmaxs();
      }
      return tensor;
   }

   @Override
   public NDArray columnArgmins() {
      Tensor tensor = new Tensor(Shape.shape(kernels(), channels(), 1, columns()));
      for (int i = 0; i < slices.length; i++) {
         tensor.slices[i] = slices[i].columnArgmins();
      }
      return tensor;
   }

   @Override
   public NDArray columnMaxs() {
      Tensor tensor = new Tensor(Shape.shape(kernels(), channels(), 1, columns()));
      for (int i = 0; i < slices.length; i++) {
         tensor.slices[i] = slices[i].columnMaxs();
      }
      return tensor;
   }

   @Override
   public NDArray columnMins() {
      Tensor tensor = new Tensor(Shape.shape(kernels(), channels(), 1, columns()));
      for (int i = 0; i < slices.length; i++) {
         tensor.slices[i] = slices[i].columnMins();
      }
      return tensor;
   }

   @Override
   public NDArray columnSums() {
      Tensor tensor = new Tensor(Shape.shape(kernels(), channels(), 1, columns()));
      for (int i = 0; i < slices.length; i++) {
         tensor.slices[i] = slices[i].columnSums();
      }
      return tensor;
   }

   @Override
   public NDArray compact() {
      for (int i = 0; i < slices.length; i++) {
         slices[i].compact();
      }
      return this;
   }

   @Override
   public NDArray diag() {
      Tensor tensor = new Tensor(Shape.shape(kernels(), channels(), rows(), columns()));
      for (int i = 0; i < slices.length; i++) {
         tensor.slices[i] = slices[i].diag();
      }
      return tensor;
   }

   @Override
   public NDArray div(NDArray rhs) {
      check(rhs.shape);
      Tensor tensor = new Tensor(shape);
      for (int i = 0; i < slices.length; i++) {
         tensor.slices[i] = slices[i].div(rhs.slice(i));
      }
      return tensor;
   }

   @Override
   public NDArray div(double value) {
      Tensor tensor = new Tensor(shape);
      for (int i = 0; i < slices.length; i++) {
         tensor.slices[i] = slices[i].div(value);
      }
      return tensor;
   }

   @Override
   public NDArray divColumnVector(NDArray rhs) {
      check(rows(), rhs.shape);
      Tensor tensor = new Tensor(shape);
      for (int i = 0; i < slices.length; i++) {
         tensor.slices[i] = slices[i].divColumnVector(rhs.slice(i));
      }
      return tensor;
   }

   @Override
   public NDArray divRowVector(NDArray rhs) {
      check(rows(), rhs.shape);
      Tensor tensor = new Tensor(shape);
      for (int i = 0; i < slices.length; i++) {
         tensor.slices[i] = slices[i].divRowVector(rhs.slice(i));
      }
      return tensor;
   }

   @Override
   public NDArray divi(NDArray rhs) {
      check(rhs.shape);
      for (int i = 0; i < slices.length; i++) {
         slices[i].divi(rhs.slice(i));
      }
      return this;
   }

   @Override
   public NDArray divi(double value) {
      for (NDArray slice : slices) {
         slice.divi(value);
      }
      return this;
   }

   @Override
   public NDArray diviColumnVector(NDArray rhs) {
      check(rows(), rhs.shape);
      for (int i = 0; i < slices.length; i++) {
         slices[i].diviColumnVector(rhs.slice(i));
      }
      return this;
   }

   @Override
   public NDArray diviRowVector(NDArray rhs) {
      check(columns(), rhs.shape);
      for (int i = 0; i < slices.length; i++) {
         slices[i].diviRowVector(rhs.slice(i));
      }
      return this;
   }

   @Override
   public double dot(NDArray rhs) {
      check(columns(), rhs.shape);
      double dot = 0;
      for (int i = 0; i < slices.length; i++) {
         dot += slices[i].dot(rhs.slice(i));
      }
      return dot;
   }

   @Override
   public NDArray fill(double value) {
      for (NDArray slice : slices) {
         slice.fill(value);
      }
      return this;
   }

   @Override
   public void forEachSparse(EntryConsumer consumer) {
      for (int i = 0; i < slices.length; i++) {
         int slice = i;
         slices[i].forEachSparse((mi, v) -> consumer.apply(mi * slice, v));
      }
   }

   @Override
   public double get(long i) {
      return slices[shape.toSliceIndex(i)].get(shape.toMatrixIndex(i));
   }

   @Override
   public double get(int row, int col) {
      return slices[0].get(row, col);
   }

   @Override
   public double get(int channel, int row, int col) {
      return slices[shape.sliceIndex(0, channel)].get(row, col);
   }

   @Override
   public double get(int kernel, int channel, int row, int col) {
      return slices[shape.sliceIndex(kernel, channel)].get(row, col);
   }

   @Override
   public NDArray getColumn(int column) {
      Tensor tensor = new Tensor(Shape.shape(kernels(), channels(), rows(), 1));
      for (int i = 0; i < slices.length; i++) {
         tensor.slices[i] = slices[i].getColumn(column);
      }
      return tensor;
   }

   @Override
   public NDArray getColumns(int[] columns) {
      NDArray[] out = new NDArray[shape.sliceLength];
      for (int i = 0; i < slices.length; i++) {
         out[i] = slices[i].getColumns(columns);
      }
      return new Tensor(kernels(), channels(), out);
   }

   @Override
   public NDArray getColumns(int from, int to) {
      NDArray[] out = new NDArray[shape.sliceLength];
      for (int i = 0; i < slices.length; i++) {
         out[i] = slices[i].getColumns(from, to);
      }
      return new Tensor(kernels(), channels(), out);
   }

   @Override
   public NDArray getRow(int row) {
      NDArray[] out = new NDArray[shape.sliceLength];
      for (int i = 0; i < slices.length; i++) {
         out[i] = slices[i].getRow(row);
      }
      return new Tensor(kernels(), channels(), out);
   }

   @Override
   public NDArray getRows(int[] rows) {
      NDArray[] out = new NDArray[shape.sliceLength];
      for (int i = 0; i < slices.length; i++) {
         out[i] = slices[i].getRows(rows);
      }
      return new Tensor(kernels(), channels(), out);
   }

   @Override
   public NDArray getRows(int from, int to) {
      NDArray[] out = new NDArray[shape.sliceLength];
      for (int i = 0; i < slices.length; i++) {
         out[i] = slices[i].getRows(from, to);
      }
      return new Tensor(kernels(), channels(), out);
   }

   @Override
   public NDArray getSubMatrix(int fromRow, int toRow, int fromCol, int toCol) {
      NDArray[] out = new NDArray[shape.sliceLength];
      for (int i = 0; i < slices.length; i++) {
         out[i] = slices[i].getSubMatrix(fromRow, toRow, fromCol, toCol);
      }
      return new Tensor(kernels(), channels(), out);
   }

   @Override
   public boolean isDense() {
      return slices[0].isDense();
   }

   @Override
   public NDArray map(DoubleUnaryOperator operator) {
      NDArray[] out = new NDArray[shape.sliceLength];
      for (int i = 0; i < slices.length; i++) {
         out[i] = slices[i].map(operator);
      }
      return new Tensor(kernels(), channels(), out);
   }

   @Override
   public NDArray map(double value, DoubleBinaryOperator operator) {
      NDArray[] out = new NDArray[shape.sliceLength];
      for (int i = 0; i < slices.length; i++) {
         out[i] = slices[i].map(value, operator);
      }
      return new Tensor(kernels(), channels(), out);
   }

   @Override
   public NDArray map(NDArray rhs, DoubleBinaryOperator operator) {
      check(rhs.shape);
      NDArray[] out = new NDArray[shape.sliceLength];
      for (int i = 0; i < slices.length; i++) {
         out[i] = slices[i].map(rhs.slice(i), operator);
      }
      return new Tensor(kernels(), channels(), out);
   }

   @Override
   public NDArray mapColumn(NDArray rhs, DoubleBinaryOperator operator) {
      check(rows(), rhs.shape);
      NDArray[] out = new NDArray[shape.sliceLength];
      for (int i = 0; i < slices.length; i++) {
         out[i] = slices[i].mapColumn(rhs.slice(i), operator);
      }
      return new Tensor(kernels(), channels(), out);
   }

   @Override
   public NDArray mapRow(NDArray rhs, DoubleBinaryOperator operator) {
      check(columns(), rhs.shape);
      NDArray[] out = new NDArray[shape.sliceLength];
      for (int i = 0; i < slices.length; i++) {
         out[i] = slices[i].mapRow(rhs.slice(i), operator);
      }
      return new Tensor(kernels(), channels(), out);
   }

   @Override
   public NDArray mapi(DoubleUnaryOperator operator) {
      for (NDArray slice : slices) {
         slice.mapi(operator);
      }
      return this;
   }

   @Override
   public NDArray mapi(double value, DoubleBinaryOperator operator) {
      for (NDArray slice : slices) {
         slice.mapi(value, operator);
      }
      return this;
   }

   @Override
   public NDArray mapi(NDArray rhs, DoubleBinaryOperator operator) {
      check(rhs.shape);
      for (int i = 0; i < slices.length; i++) {
         slices[i].mapi(rhs.slice(i), operator);
      }
      return this;
   }

   @Override
   public NDArray mapiColumn(NDArray rhs, DoubleBinaryOperator operator) {
      check(rows(), rhs.shape);
      for (int i = 0; i < slices.length; i++) {
         slices[i].mapiColumn(rhs.slice(i), operator);
      }
      return this;
   }

   @Override
   public NDArray mapiRow(NDArray rhs, DoubleBinaryOperator operator) {
      check(columns(), rhs.shape);
      for (int i = 0; i < slices.length; i++) {
         slices[i].mapiRow(rhs.slice(i), operator);
      }
      return this;
   }

   @Override
   public double max() {
      return Stream.of(slices).mapToDouble(NDArray::max).max().orElse(Double.NEGATIVE_INFINITY);
   }

   @Override
   public double min() {
      return Stream.of(slices).mapToDouble(NDArray::min).min().orElse(Double.POSITIVE_INFINITY);
   }

   @Override
   public NDArray mmul(NDArray rhs) {
      check(rhs.shape);
      NDArray[] out = new NDArray[shape.sliceLength];
      for (int i = 0; i < slices.length; i++) {
         out[i] = slices[i].mmul(rhs.slice(i));
      }
      return new Tensor(kernels(), channels(), out);
   }

   @Override
   public NDArray mul(NDArray rhs) {
      check(rhs.shape);
      NDArray[] out = new NDArray[shape.sliceLength];
      for (int i = 0; i < slices.length; i++) {
         out[i] = slices[i].mul(rhs.slice(i));
      }
      return new Tensor(kernels(), channels(), out);
   }

   @Override
   public NDArray mul(double value) {
      NDArray[] out = new NDArray[shape.sliceLength];
      for (int i = 0; i < slices.length; i++) {
         out[i] = slices[i].mul(value);
      }
      return new Tensor(kernels(), channels(), out);
   }

   @Override
   public NDArray mulColumnVector(NDArray rhs) {
      check(rows(), rhs.shape);
      NDArray[] out = new NDArray[shape.sliceLength];
      for (int i = 0; i < slices.length; i++) {
         out[i] = slices[i].mulColumnVector(rhs.slice(i));
      }
      return new Tensor(kernels(), channels(), out);
   }

   @Override
   public NDArray mulRowVector(NDArray rhs) {
      check(columns(), rhs.shape);
      NDArray[] out = new NDArray[shape.sliceLength];
      for (int i = 0; i < slices.length; i++) {
         out[i] = slices[i].mulRowVector(rhs.slice(i));
      }
      return new Tensor(kernels(), channels(), out);
   }

   @Override
   public NDArray muli(NDArray rhs) {
      check(rhs.shape);
      for (int i = 0; i < slices.length; i++) {
         slices[i].muli(rhs.slice(i));
      }
      return this;
   }

   @Override
   public NDArray muli(double value) {
      for (NDArray slice : slices) {
         slice.muli(value);
      }
      return this;
   }

   @Override
   public NDArray muliColumnVector(NDArray rhs) {
      check(rows(), rhs.shape);
      for (int i = 0; i < slices.length; i++) {
         slices[i].muliColumnVector(rhs.slice(i));
      }
      return this;
   }

   @Override
   public NDArray muliRowVector(NDArray rhs) {
      check(columns(), rhs.shape);
      for (int i = 0; i < slices.length; i++) {
         slices[i].muliRowVector(rhs.slice(i));
      }
      return this;
   }

   @Override
   public double norm1() {
      return Stream.of(slices).mapToDouble(NDArray::norm1).sum();
   }

   @Override
   public double norm2() {
      return Math.sqrt(Stream.of(slices).mapToDouble(NDArray::sumOfSquares).sum());
   }

   @Override
   public NDArray pivot() {
      NDArray[] out = new NDArray[shape.sliceLength];
      for (int i = 0; i < slices.length; i++) {
         out[i] = slices[i].pivot();
      }
      return new Tensor(kernels(), channels(), out);
   }

   @Override
   public NDArray rdiv(NDArray rhs) {
      return null;
   }

   @Override
   public NDArray rdiv(double value) {
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
   public NDArray rdivi(double value) {
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
   public NDArray reshape(int... dims) {
      return null;
   }

   @Override
   public NDArray rowArgmaxs() {
      return null;
   }

   @Override
   public NDArray rowArgmins() {
      return null;
   }

   @Override
   public NDArray rowMaxs() {
      return null;
   }

   @Override
   public NDArray rowMins() {
      return null;
   }

   @Override
   public NDArray rowSums() {
      return null;
   }

   @Override
   public NDArray rsub(NDArray rhs) {
      return null;
   }

   @Override
   public NDArray rsub(double value) {
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
   public NDArray rsubi(double value) {
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
   public NDArray select(DoublePredicate predicate) {
      return null;
   }

   @Override
   public NDArray select(NDArray rhs) {
      return null;
   }

   @Override
   public NDArray selecti(DoublePredicate predicate) {
      return null;
   }

   @Override
   public NDArray selecti(NDArray rhs) {
      return null;
   }

   @Override
   public NDArray set(long i, double value) {
      return null;
   }

   @Override
   public NDArray set(int row, int col, double value) {
      return null;
   }

   @Override
   public NDArray set(int channel, int row, int col, double value) {
      return null;
   }

   @Override
   public NDArray set(int kernel, int channel, int row, int col, double value) {
      return null;
   }

   @Override
   public NDArray setColumn(int i, NDArray array) {
      return null;
   }

   @Override
   public NDArray setRow(int i, NDArray array) {
      return null;
   }

   @Override
   public NDArray setSlice(int slice, NDArray array) {
      return null;
   }

   @Override
   public Shape shape() {
      return null;
   }

   @Override
   public long size() {
      return 0;
   }

   @Override
   public NDArray slice(int slice) {
      return null;
   }

   @Override
   public NDArray sliceArgmaxs() {
      return null;
   }

   @Override
   public NDArray sliceArgmins() {
      return null;
   }

   @Override
   public NDArray sliceDot(NDArray rhs) {
      return null;
   }

   @Override
   public NDArray sliceMaxs() {
      return null;
   }

   @Override
   public NDArray sliceMeans() {
      return null;
   }

   @Override
   public NDArray sliceMins() {
      return null;
   }

   @Override
   public NDArray sliceNorm1() {
      return null;
   }

   @Override
   public NDArray sliceNorm2() {
      return null;
   }

   @Override
   public NDArray sliceSumOfSquares() {
      return null;
   }

   @Override
   public NDArray sliceSums() {
      return null;
   }

   @Override
   public NDArray sub(NDArray rhs) {
      return null;
   }

   @Override
   public NDArray sub(double value) {
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
   public NDArray subi(double value) {
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
   public double sum() {
      return 0;
   }

   @Override
   public double sumOfSquares() {
      return 0;
   }

   @Override
   public NDArray test(DoublePredicate predicate) {
      return null;
   }

   @Override
   public NDArray test(NDArray rhs, DoubleBinaryPredicate predicate) {
      return null;
   }

   @Override
   public NDArray testi(DoublePredicate predicate) {
      return null;
   }

   @Override
   public NDArray testi(NDArray rhs, DoubleBinaryPredicate predicate) {
      return null;
   }

   @Override
   public double[] toDoubleArray() {
      return new double[0];
   }

   @Override
   public DoubleMatrix[] toDoubleMatrix() {
      return new DoubleMatrix[0];
   }

   @Override
   public NDArray unitize() {
      return null;
   }

   @Override
   public NDArray zeroLike() {
      return null;
   }
}//END OF Tensor
