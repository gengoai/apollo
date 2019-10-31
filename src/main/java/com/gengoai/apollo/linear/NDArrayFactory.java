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

import com.gengoai.Validation;
import com.gengoai.config.Config;
import org.jblas.DoubleMatrix;

import java.util.Arrays;
import java.util.Collection;
import java.util.Random;

import static com.gengoai.apollo.linear.Shape.shape;

/**
 * Factories for creating NDArrays
 *
 * @author David B. Bracewell
 */
public enum NDArrayFactory {
   /**
    * The default factory which checks the Config property <code>NDArrayFactory.default</code> to determine if the
    * default factory to use is  <code>SPARSE</code> or <code>DENSE</code>. If the config property is not, set it will
    * use <code>DENSE</code>.
    */
   ND {
      private volatile NDArrayFactory factory;

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
    * Dense NDArrays backed by JBlas
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
      public NDArray array(double[] data) {
         return new DenseMatrix(new DoubleMatrix(data));
      }

      @Override
      public NDArray array(int rows, int columns, double[] data) {
         return new DenseMatrix(new DoubleMatrix(rows, columns, data));
      }

      @Override
      public NDArray array(double[][] data) {
         return new DenseMatrix(new DoubleMatrix(data));
      }

      @Override
      public NDArray columnVector(double[] data) {
         return new DenseMatrix(new DoubleMatrix(data));
      }

      @Override
      public NDArray rowVector(double[] data) {
         return new DenseMatrix(new DoubleMatrix(1, data.length, data));
      }


   },
   /**
    * Sparse NDArrays (expect them to be 3-10x slower, but more space efficient).
    */
   SPARSE {
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

   /**
    * Generates an NDArray from the given slices where each slice represents a single channel.
    *
    * @param slices the slices
    * @return the NDArray
    */
   public NDArray array(NDArray[] slices) {
      return array(0, slices.length, slices);
   }

   /**
    * Creates an NDArray out of the array of slices with the given number of kernels and channels.
    *
    * @param kernels  the number of kernels
    * @param channels the number of channels
    * @param slices   the slices
    * @return the NDArray
    */
   public NDArray array(int kernels, int channels, NDArray[] slices) {
      Validation.checkArgument(kernels * channels == slices.length,
                               () -> "Invalid Slice Length: " + (kernels * channels) + " != " + slices.length);
      return new Tensor(kernels, channels, slices);
   }

   /**
    * Creates a zero-valued NDArray with the given dimensions.
    *
    * @param dims the shape of the NDArray
    * @return the NDArray
    */
   public NDArray array(int... dims) {
      return array(new Shape(dims));
   }

   /**
    * Creates a 1-d column vector NDArray of the given data.
    *
    * @param data the data
    * @return the NDArray
    */
   public NDArray array(double[] data) {
      return columnVector(data);
   }

   /**
    * Creates a 2-d NDArray with the given number of rows and columns using the given data.
    *
    * @param rows    the number of rows
    * @param columns the number of columns
    * @param data    the data
    * @return the NDArray
    */
   public NDArray array(int rows, int columns, double[] data) {
      Validation.checkArgument(rows * columns == data.length,
                               () -> "Invalid Length: " + (rows * columns) + " != " + data.length);
      NDArray out = array(rows, columns);
      for (int i = 0; i < data.length; i++) {
         out.set(i, data[i]);
      }
      return out;
   }

   /**
    * Creates a 2-d NDArray from the given data.
    *
    * @param data the data
    * @return the NDArray
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
    * Creates a zero-valued NDArray with the given shape.
    *
    * @param shape the shape
    * @return the NDArray
    */
   public abstract NDArray array(Shape shape);

   /**
    * Creates an NDArray with the given shape initialized using the given initializer.
    *
    * @param initializer the initializer to use to set initial values of the NDArray.
    * @param shape       the shape
    * @return the NDArray
    */
   public NDArray array(Shape shape, NDArrayInitializer initializer) {
      NDArray array = array(shape);
      initializer.accept(array);
      return array;
   }

   /**
    * Creates a 1-d column vector NDArray from the given data.
    *
    * @param data the data
    * @return the NDArray
    */
   public NDArray columnVector(double[] data) {
      NDArray vector = array(data.length, 1);
      for (int i = 0; i < data.length; i++) {
         vector.set(i, data[i]);
      }
      return vector;
   }

   /**
    * Creates an NDArray of the given shape with the given initial value
    *
    * @param value the initial value for all elements in the NDArray
    * @param shape the shape
    * @return the NDArray
    */
   public NDArray constant(Shape shape, double value) {
      return array(shape).fill(value);
   }

   /**
    * Creates an empty NDArray
    *
    * @return the NDArray
    */
   public NDArray empty() {
      return array(Shape.empty());
   }

   /**
    * Creates an identity matrix.
    *
    * @param size the number of rows/columns
    * @return the NDArray
    */
   public NDArray eye(int size) {
      NDArray ndArray = array(size, size);
      for (int i = 0; i < size; i++) {
         ndArray.set(i, i, 1);
      }
      return ndArray;
   }


   /**
    * Stacks the given NDArray horizontal, i.e. concatenates on the column axis.
    *
    * @param arrays the NDArrays
    * @return the NDArray
    */
   public NDArray hstack(NDArray... arrays) {
      return hstack(Arrays.asList(arrays));
   }


   /**
    * Stacks the given NDArray horizontal, i.e. concatenates on the column axis.
    *
    * @param arrays the NDArrays
    * @return the NDArray
    */
   public NDArray hstack(Collection<NDArray> arrays) {
      if (arrays.size() == 0) {
         return empty();
      }
      Shape shape = arrays.iterator().next().shape();
      NDArray toReturn = array(shape.matrixLength, arrays.size());
      int globalAxisIndex = 0;
      for (NDArray array : arrays) {
         toReturn.setColumn(globalAxisIndex, array);
         globalAxisIndex++;
      }
      return toReturn;
   }

   /**
    * Creates an NDArray with given shape with all values equal to 1
    *
    * @param dims the dimension of the NDArray
    * @return the NDArray
    */
   public NDArray ones(int... dims) {
      return constant(shape(dims), 1);
   }

   /**
    * Creates an NDArray with given shape with all values equal to 1
    *
    * @param shape the dimension of the NDArray
    * @return the NDArray
    */
   public NDArray ones(Shape shape) {
      return constant(shape, 1);
   }


   /**
    * Creates an NDArray with given shape with all values set randomly
    *
    * @param dims the dimension of the NDArray
    * @return the NDArray
    */
   public NDArray rand(int... dims) {
      return array(shape(dims), NDArrayInitializer.rand);
   }

   /**
    * Creates an NDArray with given shape with all values set randomly
    *
    * @param shape the dimension of the NDArray
    * @return the NDArray
    */
   public NDArray rand(Shape shape) {
      return array(shape, NDArrayInitializer.rand);
   }

   /**
    * Creates an NDArray with given shape with all values set to a random gaussian.
    *
    * @param dims the dimension of the NDArray
    * @return the NDArray
    */
   public NDArray randn(int... dims) {
      return array(Shape.shape(dims), NDArrayInitializer.randn(new Random()));
   }

   /**
    * Creates an NDArray with given shape with all values set to a random gaussian.
    *
    * @param shape the dimension of the NDArray
    * @return the NDArray
    */
   public NDArray randn(Shape shape) {
      return array(shape, NDArrayInitializer.randn(new Random()));
   }

   /**
    * Creates a 1-d row vector NDArray from the given data.
    *
    * @param data the data
    * @return the NDArray
    */
   public NDArray rowVector(double[] data) {
      NDArray vector = array(1, data.length);
      for (int i = 0; i < data.length; i++) {
         vector.set(i, data[i]);
      }
      return vector;
   }

   /**
    * Creates a scalar-valued NDArray with the given value
    *
    * @param value the value
    * @return the NDArray
    */
   public NDArray scalar(double value) {
      NDArray ndArray = array();
      ndArray.set(0, value);
      return ndArray;
   }

   /**
    * Creates an NDArray with the given shape initialized using a uniform random distribution between the
    * <code>lower</code> and <code>upper</code> bounds.
    *
    * @param lower the lower bounds of element values
    * @param upper the upper bounds of element values
    * @param shape the shape
    * @return the NDArray
    */
   public NDArray uniform(Shape shape, int lower, int upper) {
      return array(shape, NDArrayInitializer.rand(lower, upper));
   }

   /**
    * Stacks the given NDArray vertically, i.e. concatenates on the row axis.
    *
    * @param arrays the NDArrays
    * @return the NDArray
    */
   public NDArray vstack(NDArray... arrays) {
      return vstack(Arrays.asList(arrays));
   }

   /**
    * Stacks the given NDArray vertically, i.e. concatenates on the row axis.
    *
    * @param arrays the NDArrays
    * @return the NDArray
    */
   public NDArray vstack(Collection<NDArray> arrays) {
      if (arrays.size() == 0) {
         return empty();
      }
      Shape shape = arrays.iterator().next().shape();
      NDArray toReturn = array(arrays.size(), shape.matrixLength);
      int globalAxisIndex = 0;
      for (NDArray array : arrays) {
         toReturn.setRow(globalAxisIndex, array);
         globalAxisIndex++;
      }
      return toReturn;
   }

}//END OF NDArrayFactory
