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
import com.gengoai.Validation;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Objects;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * The type Shape.
 *
 * @author David B. Bracewell
 */
public class Shape implements Serializable, Copyable<Shape> {
   private static final long serialVersionUID = 1L;
   /**
    * The Shape.
    */
   final int[] shape;
   public final int matrixLength;
   public final int sliceLength;


   public static Shape empty() {
      Shape s = new Shape();
      s.shape[2] = 0;
      s.shape[3] = 0;
      return s;
   }

   public static Shape shape(int... dims) {
      return new Shape(dims);
   }


   public int sliceIndex(int kernel, int channel) {
      return kernel + (shape[0] * channel);
   }

   public int matrixIndex(int row, int column) {
      return row + (shape[2] * column);
   }


   /**
    * Instantiates a new Shape.
    *
    * @param dimensions the dimensions
    */
   public Shape(int... dimensions) {
      this.shape = new int[4];
      if (dimensions != null && dimensions.length > 0) {
         System.arraycopy(dimensions, 0, shape, shape.length - dimensions.length, dimensions.length);
         this.shape[2] = Math.max(1, this.shape[2]);
         this.shape[3] = Math.max(1, this.shape[3]);
         this.sliceLength = Math.max(1, shape[0] * shape[1]);
         this.matrixLength = Math.max(1, shape[2] * shape[3]);

      } else {
         this.shape[2] = 1;
         this.shape[3] = 1;
         this.sliceLength = 1;
         this.matrixLength = 1;
      }
      Validation.checkArgument(shape[0] >= 0, "Invalid Kernel: " + shape[0]);
      Validation.checkArgument(shape[1] >= 0, "Invalid Channel: " + shape[1]);
      Validation.checkArgument(shape[2] >= 0, "Invalid Row: " + shape[2]);
      Validation.checkArgument(shape[3] >= 0, "Invalid Column: " + shape[3]);
   }

   /**
    * Channels int.
    *
    * @return the int
    */
   public int channels() {
      return shape[1];
   }

   /**
    * Columns int.
    *
    * @return the int
    */
   public int columns() {
      return shape[3];
   }

   @Override
   public Shape copy() {
      return new Shape(this.shape);
   }

   /**
    * Dim int.
    *
    * @param index the index
    * @return the int
    */
   public int dim(int index) {
      if (index > shape.length || index < 0) {
         return 0;
      }
      return shape[index];
   }

   @Override
   public boolean equals(Object obj) {
      if (this == obj) {return true;}
      if (obj == null || getClass() != obj.getClass()) {return false;}
      final Shape other = (Shape) obj;
      return Objects.deepEquals(this.shape, other.shape);
   }

   @Override
   public int hashCode() {
      return Arrays.hashCode(shape);
   }

   /**
    * Is matrix boolean.
    *
    * @return the boolean
    */
   public boolean isMatrix() {
      return (shape[0] == 0 && shape[1] == 0)
                && (shape[2] > 0 && shape[3] > 0);
   }

   /**
    * Is scalar boolean.
    *
    * @return the boolean
    */
   public boolean isScalar() {
      return shape[0] == 0 && shape[1] == 0 && shape[2] == 1 && shape[3] == 1;
   }

   /**
    * Is tensor boolean.
    *
    * @return the boolean
    */
   public boolean isTensor() {
      return shape[0] > 0 || shape[1] > 0;
   }

   /**
    * Is vector boolean.
    *
    * @return the boolean
    */
   public boolean isVector() {
      return (shape[0] == 0 && shape[1] == 0)
                && (shape[2] > 0 ^ shape[3] > 0);
   }

   public boolean isRowVector() {
      return (shape[0] == 0 && shape[1] == 0)
                && (shape[2] == 1 && shape[3] > 1);
   }

   public boolean isColumnVector() {
      return (shape[0] == 0 && shape[1] == 0)
                && (shape[2] > 1 && shape[3] == 1);
   }

   public boolean isSquare() {
      return shape[0] == 0 && shape[1] == 0 && shape[2] == shape[3];
   }

   /**
    * Kernels int.
    *
    * @return the int
    */
   public int kernels() {
      return shape[0];
   }

   /**
    * Order int.
    *
    * @return the int
    */
   public int order() {
      int order = 0;
      for (int i1 : shape) {
         order += i1 > 1 ? 1 : 0;
      }
      return order;
   }

   /**
    * Reshape.
    *
    * @param dimensions the dimensions
    */
   public void reshape(int... dimensions) {
      Shape out = new Shape(dimensions);
      if (sliceLength != out.sliceLength) {
         throw new IllegalArgumentException("Invalid slice length: " + sliceLength + " != " + out.sliceLength);
      }
      if (matrixLength != out.matrixLength) {
         throw new IllegalArgumentException("Invalid matrix length: " + matrixLength + " != " + out.matrixLength);
      }
      System.arraycopy(out.shape, 0, shape, 0, shape.length);
   }

   /**
    * Rows int.
    *
    * @return the int
    */
   public int rows() {
      return shape[2];
   }

   @Override
   public String toString() {
      return "(" + IntStream.of(shape)
                            .filter(i -> i > 0)
                            .mapToObj(Integer::toString)
                            .collect(Collectors.joining(", ")) + ")";
   }


}//END OF Shape
