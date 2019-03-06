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

import com.gengoai.Validation;
import com.gengoai.apollo.linear.Shape;
import org.jblas.FloatMatrix;

/**
 * @author David B. Bracewell
 */
public class DenseNDArray extends NDArray {

   private final FloatMatrix[] slices;

   public DenseNDArray(Shape shape) {
      super(shape);
      this.slices = new FloatMatrix[Math.max(shape.columns() * shape.kernels(), 1)];
      for (int i = 0; i < this.slices.length; i++) {
         this.slices[i] = new FloatMatrix(shape.rows(), shape.columns());
      }
   }

   protected DenseNDArray(FloatMatrix fm) {
      super(new Shape(fm.rows, fm.columns));
      this.slices = new FloatMatrix[]{fm};
   }

   public NDArray getMatrix(int sliceIndex) {
      if (shape.isMatrix()) {
         return new DenseNDArray(slices[0]);
      }
      return new DenseNDArray(slices[sliceIndex]);
   }

   @Override
   public NDArray add(NDArray rhs) {
      Validation.checkArgument(rhs.shape.order() <= shape.order());
      if (shape.order() == rhs.shape.order()) {
         return null;
      }
      switch (rhs.shape.order()) {
         case 2:
            return null;
         case 1:
            if (rhs.shape.isColumnVector()) {

            } else {

            }
         default:
            return null;
      }
   }

   @Override
   public NDArray add(double rhs) {
      return null;
   }

   @Override
   public NDArray addColumnVector(NDArray rhs) {
      return null;
   }

   @Override
   public NDArray addRowVector(NDArray rhs) {
      return null;
   }

   @Override
   public NDArray addi(NDArray rhs) {
      return null;
   }

   @Override
   public NDArray addi(double rhs) {
      return null;
   }

   @Override
   public NDArray addiColumnVector(NDArray rhs) {
      return null;
   }

   @Override
   public NDArray addiRowVector(NDArray rhs) {
      return null;
   }

   @Override
   public NDArray div(NDArray rhs) {
      return null;
   }

   @Override
   public NDArray div(double rhs) {
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
   public NDArray divi(double rhs) {
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
   public NDArray get(int i) {
      return null;
   }

   @Override
   public NDArray get(int i, int j) {
      return null;
   }

   @Override
   public NDArray get(int i, int j, int k) {
      return null;
   }

   @Override
   public NDArray get(int i, int j, int k, int l) {
      return null;
   }

   @Override
   public NDArray mul(NDArray rhs) {
      return null;
   }

   @Override
   public NDArray mul(double rhs) {
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
   public NDArray muli(double rhs) {
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
   public NDArray rdiv(NDArray rhs) {
      return null;
   }

   @Override
   public NDArray rdiv(double rhs) {
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
   public NDArray rdivi(double rhs) {
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
   public NDArray rsub(NDArray rhs) {
      return null;
   }

   @Override
   public NDArray rsub(double rhs) {
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
   public NDArray rsubi(double rhs) {
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
   public NDArray set(int i, double value) {
      return null;
   }

   @Override
   public NDArray set(int i, int j, double value) {
      return null;
   }

   @Override
   public NDArray set(int i, int j, int k, double value) {
      return null;
   }

   @Override
   public NDArray set(int i, int j, int k, int l, double value) {
      return null;
   }

   @Override
   public NDArray sub(NDArray rhs) {
      return null;
   }

   @Override
   public NDArray sub(double rhs) {
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
   public NDArray subi(double rhs) {
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
}//END OF DenseNDArray
