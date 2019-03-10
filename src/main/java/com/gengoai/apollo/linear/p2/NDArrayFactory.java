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
      NDArray ndArray = array(dims);
      for (int i = 0; i < ndArray.shape().sliceLength; i++) {
         NDArray slice = ndArray.slice(i);
         for (int mi = 0; mi < slice.shape().matrixLength; mi++) {
            if (slice.isDense() || Math.random() > SPARSE.getSparsity()) {
               slice.set(mi, Math.random());
            }
         }
      }
      return ndArray;
   }

   public NDArray uniform(double lower, double upper, Shape shape) {
      NDArray ndArray = array(shape);
      for (int i = 0; i < ndArray.shape().sliceLength; i++) {
         NDArray slice = ndArray.slice(i);
         for (int mi = 0; mi < slice.shape().matrixLength; mi++) {
            if (slice.isDense() || Math.random() > SPARSE.getSparsity()) {
               slice.set(mi, ThreadLocalRandom.current().nextDouble(lower, upper));
            }
         }
      }
      return ndArray;
   }

   public NDArray randn(int... dims) {
      final Random rnd = new Random();
      NDArray ndArray = array(dims);
      for (int i = 0; i < ndArray.shape().sliceLength; i++) {
         NDArray slice = ndArray.slice(i);
         for (int mi = 0; mi < slice.shape().matrixLength; mi++) {
            if (slice.isDense() || Math.random() > SPARSE.getSparsity()) {
               slice.set(mi, rnd.nextGaussian());
            }
         }
      }
      return ndArray;
   }

   public NDArray scalar(double value) {
      NDArray ndArray = array();
      ndArray.set(0, value);
      return ndArray;
   }


}//END OF NDArrayFactory
