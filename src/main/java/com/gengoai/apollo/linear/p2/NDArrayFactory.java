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

import com.gengoai.apollo.linear.Shape;
import com.gengoai.config.Config;
import org.jblas.DoubleMatrix;

import java.util.Arrays;
import java.util.Collection;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

import static com.gengoai.apollo.linear.Shape.shape;

/**
 * @author David B. Bracewell
 */
public enum NDArrayFactory {
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
         return new DenseMatrix(new DoubleMatrix(data.length, 1, data));
      }

      @Override
      public NDArray rowVector(double[] data) {
         return new DenseMatrix(new DoubleMatrix(1, data.length, data));
      }

      @Override
      public NDArray array(Shape shape) {
         return getFactory().array(shape);
      }
   },
   DENSE {
      @Override
      public NDArray array(Shape shape) {
         return new DenseMatrix(shape);
      }

   },
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
         return new SparseMatrix(shape);
      }
   };

   public NDArray eye(int size) {
      NDArray ndArray = array(size, size);
      for (int i = 0; i < size; i++) {
         ndArray.set(i, i, 1);
      }
      return ndArray;
   }

   public final NDArray empty() {
      return array(Shape.empty());
   }

   public final NDArray array(int... dims) {
      return array(new Shape(dims));
   }

   public abstract NDArray array(Shape shape);

   public NDArray constant(double value, Shape shape) {
      return array(shape).fill(value);
   }

   public double getSparsity() {
      return 0d;
   }

   public NDArray ones(int... dims) {
      return constant(1d, shape(dims));
   }

   public NDArray rand(int... dims) {
      return array(dims).mapi(v -> Math.random());
   }

   public NDArray uniform(double lower, double upper, Shape shape) {
      return array(shape).mapi(v -> ThreadLocalRandom.current().nextDouble(lower, upper));
   }

   public NDArray randn(int... dims) {
      final Random rnd = new Random();
      return array(dims).mapi(v -> rnd.nextGaussian());
   }

   public NDArray scalar(double value) {
      NDArray ndArray = array();
      ndArray.set(0, value);
      return ndArray;
   }

   public NDArray vstack(NDArray... rows) {
      return vstack(Arrays.asList(rows));
   }

   public NDArray vstack(Collection<NDArray> rows) {
      if (rows.size() == 0) {
         return empty();
      }
      Shape shape = rows.iterator().next().shape();
      NDArray toReturn = array(rows.size(), shape.matrixLength);
      int globalAxisIndex = 0;
      for (NDArray array : rows) {
         toReturn.setRow(globalAxisIndex, array);
      }
      return toReturn;
   }

   public NDArray hstack(NDArray... columns) {
      return hstack(Arrays.asList(columns));
   }

   public NDArray hstack(Collection<NDArray> columns) {
      if (columns.size() == 0) {
         return empty();
      }
      Shape shape = columns.iterator().next().shape();
      NDArray toReturn = array(shape.matrixLength, columns.size());
      int globalAxisIndex = 0;
      for (NDArray array : columns) {
         toReturn.setColumn(globalAxisIndex, array);
      }
      return toReturn;
   }

   public NDArray columnVector(double[] data) {
      NDArray vector = array(data.length, 1);
      for (int i = 0; i < data.length; i++) {
         vector.set(i, data[i]);
      }
      return vector;
   }

   public NDArray rowVector(double[] data) {
      NDArray vector = array(1, data.length);
      for (int i = 0; i < data.length; i++) {
         vector.set(i, data[i]);
      }
      return vector;
   }

}//END OF NDArrayFactory
