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

import com.gengoai.config.Config;
import org.jblas.DoubleMatrix;

import java.util.Arrays;
import java.util.Collection;
import java.util.Random;

import static com.gengoai.apollo.linear.Shape.shape;

/**
 * The enum Nd array factory.
 *
 * @author David B. Bracewell
 */
public enum NDArrayFactory {
   /**
    * The Nd.
    */
   ND {
      private NDArrayFactory factory;

      private NDArrayFactory getFactory() {
         if (factory == null) {
            synchronized (ND) {
               if (factory == null) {
                  factory = Config.get("NDArrayFactory.default").as(NDArrayFactory.class, DENSE);
               }
            }
         }
         return factory;
      }

      @Override
      public NDArray columnVector(double[] data) {
         return getFactory().columnVector(data);
      }

      @Override
      public NDArray rowVector(double[] data) {
         return getFactory().rowVector(data);
      }

      @Override
      public NDArray array(Shape shape) {
         return getFactory().array(shape);
      }
   },
   /**
    * The Dense.
    */
   DENSE {
      @Override
      public NDArray array(Shape shape) {
         if (shape.isTensor()) {
            Tensor tensor = new Tensor(shape);
            for (int i = 0; i < shape.sliceLength; i++) {
               tensor.slices[i] = new DenseMatrix(shape.rows(), shape.columns());
            }
            return tensor;
         }
         return new DenseMatrix(shape);
      }


      @Override
      public NDArray array(double[][] data) {
         return new DenseMatrix(new DoubleMatrix(data));
      }

   },
   /**
    * The Sparse.
    */
   SPARSE {
      private Double sparsity = null;

      @Override
      public double getSparsity() {
         if (sparsity == null) {
            synchronized (SPARSE) {
               if (sparsity == null) {
                  sparsity = Config.get("SparseFactory.sparsity").asDouble(0.9);
               }
            }
         }
         return sparsity;
      }

      @Override
      public NDArray array(Shape shape) {
         if (shape.isTensor()) {
            Tensor tensor = new Tensor(shape);
            for (int i = 0; i < shape.sliceLength; i++) {
               tensor.slices[i] = new SparseMatrix(shape.rows(), shape.columns());
            }
            return tensor;
         }
         return new SparseMatrix(shape);
      }

   };

   public final NDArray array(NDArray[] slices) {
      return array(0, slices.length, slices);
   }

   public NDArray array(int kernels, int channels, NDArray[] slices) {
      return new Tensor(kernels, channels, slices);
   }

   /**
    * Array nd array.
    *
    * @param dims the dims
    * @return the nd array
    */
   public final NDArray array(int... dims) {
      return array(new Shape(dims));
   }

   /**
    * Array nd array.
    *
    * @param data the data
    * @return the nd array
    */
   public final NDArray array(double[] data) {
      return rowVector(data);
   }

   /**
    * Array nd array.
    *
    * @param rows    the rows
    * @param columns the columns
    * @param data    the data
    * @return the nd array
    */
   public final NDArray array(int rows, int columns, double[] data) {
      NDArray out = array(rows, columns);
      for (int i = 0; i < data.length; i++) {
         out.set(i, data[i]);
      }
      return out;
   }

   /**
    * Array nd array.
    *
    * @param data the data
    * @return the nd array
    */
   public NDArray array(double[][] data) {
      if (data.length == 0) {
         return empty();
      }
      NDArray array = array(data.length, data[0].length);
      for (int i = 0; i < data.length; i++) {
         array.setRow(i, rowVector(data[i]));
      }
      return array;
   }

   /**
    * Array nd array.
    *
    * @param shape the shape
    * @return the nd array
    */
   public abstract NDArray array(Shape shape);

   /**
    * Array nd array.
    *
    * @param initializer the initializer
    * @param shape       the shape
    * @return the nd array
    */
   public NDArray array(NDArrayInitializer initializer, Shape shape) {
      NDArray array = array(shape);
      initializer.accept(array);
      return array;
   }

   /**
    * Column vector nd array.
    *
    * @param data the data
    * @return the nd array
    */
   public NDArray columnVector(double[] data) {
      NDArray vector = array(data.length, 1);
      for (int i = 0; i < data.length; i++) {
         vector.set(i, data[i]);
      }
      return vector;
   }

   /**
    * Constant nd array.
    *
    * @param value the value
    * @param shape the shape
    * @return the nd array
    */
   public NDArray constant(double value, Shape shape) {
      return array(shape).fill(value);
   }

   /**
    * Empty nd array.
    *
    * @return the nd array
    */
   public final NDArray empty() {
      return array(Shape.empty());
   }

   /**
    * Eye nd array.
    *
    * @param size the size
    * @return the nd array
    */
   public NDArray eye(int size) {
      NDArray ndArray = array(size, size);
      for (int i = 0; i < size; i++) {
         ndArray.set(i, i, 1);
      }
      return ndArray;
   }

   /**
    * Gets sparsity.
    *
    * @return the sparsity
    */
   public double getSparsity() {
      return 0d;
   }

   /**
    * Hstack nd array.
    *
    * @param columns the columns
    * @return the nd array
    */
   public NDArray hstack(NDArray... columns) {
      return hstack(Arrays.asList(columns));
   }

   /**
    * Hstack nd array.
    *
    * @param columns the columns
    * @return the nd array
    */
   public NDArray hstack(Collection<NDArray> columns) {
      if (columns.size() == 0) {
         return empty();
      }
      Shape shape = columns.iterator().next().shape();
      NDArray toReturn = array(shape.matrixLength, columns.size());
      int globalAxisIndex = 0;
      for (NDArray array : columns) {
         toReturn.setColumn(globalAxisIndex, array);
         globalAxisIndex++;
      }
      return toReturn;
   }

   /**
    * Ones nd array.
    *
    * @param dims the dims
    * @return the nd array
    */
   public NDArray ones(int... dims) {
      return constant(1d, shape(dims));
   }

   /**
    * Rand nd array.
    *
    * @param dims the dims
    * @return the nd array
    */
   public NDArray rand(int... dims) {
      return array(NDArrayInitializer.rand, shape(dims));
   }

   /**
    * Randn nd array.
    *
    * @param dims the dims
    * @return the nd array
    */
   public NDArray randn(int... dims) {
      return array(NDArrayInitializer.randn(new Random()), Shape.shape(dims));
   }

   /**
    * Row vector nd array.
    *
    * @param data the data
    * @return the nd array
    */
   public NDArray rowVector(double[] data) {
      NDArray vector = array(1, data.length);
      for (int i = 0; i < data.length; i++) {
         vector.set(i, data[i]);
      }
      return vector;
   }

   /**
    * Scalar nd array.
    *
    * @param value the value
    * @return the nd array
    */
   public NDArray scalar(double value) {
      NDArray ndArray = array();
      ndArray.set(0, value);
      return ndArray;
   }

   /**
    * Uniform nd array.
    *
    * @param lower the lower
    * @param upper the upper
    * @param shape the shape
    * @return the nd array
    */
   public NDArray uniform(int lower, int upper, Shape shape) {
      return array(NDArrayInitializer.rand(lower, upper), shape);
   }

   /**
    * Vstack nd array.
    *
    * @param rows the rows
    * @return the nd array
    */
   public NDArray vstack(NDArray... rows) {
      return vstack(Arrays.asList(rows));
   }

   /**
    * Vstack nd array.
    *
    * @param rows the rows
    * @return the nd array
    */
   public NDArray vstack(Collection<NDArray> rows) {
      if (rows.size() == 0) {
         return empty();
      }
      Shape shape = rows.iterator().next().shape();
      NDArray toReturn = array(rows.size(), shape.matrixLength);
      int globalAxisIndex = 0;
      for (NDArray array : rows) {
         toReturn.setRow(globalAxisIndex, array);
         globalAxisIndex++;
      }
      return toReturn;
   }

}//END OF NDArrayFactory
