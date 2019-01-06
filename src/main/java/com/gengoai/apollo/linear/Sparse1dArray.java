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

import org.apache.mahout.math.map.OpenIntFloatHashMap;
import org.jblas.DoubleMatrix;
import org.jblas.FloatMatrix;

import static com.gengoai.Validation.checkArgument;
import static com.gengoai.Validation.checkElementIndex;

/**
 * @author David B. Bracewell
 */
public class Sparse1dArray extends NDArray {
   private static final long serialVersionUID = 1L;
   private final OpenIntFloatHashMap map = new OpenIntFloatHashMap();

   /**
    * Default Constructor
    *
    * @param shape The shape of the new NDArray
    * @throws IllegalArgumentException if the length of the shape array is greater than four.
    */
   protected Sparse1dArray(int[] shape) {
      super(shape);
      if (shape[2] > 1 || shape[3] > 1) {
         throw new IllegalArgumentException();
      } else if (shape[0] > 1) {
         checkArgument(shape[1] <= 1);
      }
   }

   protected Sparse1dArray(float[] data) {
      super(new int[]{1, data.length, 1, 1});
      for (int i = 0; i < data.length; i++) {
         if (data[i] != 0) {
            map.put(i, data[i]);
         }
      }
   }

   @Override
   public NDArray adjustIndexedValue(int sliceIndex, int matrixIndex, double value) {
      checkElementIndex(sliceIndex, numSlices(), "Slice");
      checkElementIndex(matrixIndex, sliceLength(), "Matrix");
      map.adjustOrPutValue(matrixIndex, (float) value, (float) value);
      return this;
   }

   @Override
   protected NDArray copyData() {
      return new Sparse1dArray(toFloatArray());
   }

   @Override
   public NDArrayFactory getFactory() {
      return null;
   }

   @Override
   public float getIndexedValue(int sliceIndex, int matrixIndex) {
      checkElementIndex(sliceIndex, numSlices(), "Slice");
      checkElementIndex(matrixIndex, sliceLength(), "Matrix");
      return map.get(matrixIndex);
   }

   @Override
   public NDArray getSlice(int index) {
      checkElementIndex(index, numSlices(), "Slice");
      return this;
   }

   @Override
   public NDArray mmul(NDArray other) {
      return null;
   }

   @Override
   public NDArray setIndexedValue(int sliceIndex, int matrixIndex, double value) {
      checkElementIndex(sliceIndex, numSlices(), "Slice");
      checkElementIndex(matrixIndex, sliceLength(), "Matrix");
      map.put(matrixIndex, (float) value);
      return this;
   }

   @Override
   protected void setSlice(int slice, NDArray newSlice) {

   }

   @Override
   public NDArray slice(int from, int to) {
      return null;
   }

   @Override
   public NDArray slice(int rowFrom, int rowTo, int colFrom, int colTo) {
      return null;
   }

   @Override
   public DoubleMatrix toDoubleMatrix() {
      return new DoubleMatrix(toDoubleArray());
   }

   @Override
   public FloatMatrix toFloatMatrix() {
      return new FloatMatrix(toFloatArray());
   }

}//END OF Sparse1dArray
