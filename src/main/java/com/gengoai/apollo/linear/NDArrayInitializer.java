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

import com.gengoai.function.SerializableConsumer;

import java.util.Random;

/**
 * @author David B. Bracewell
 */
public interface NDArrayInitializer extends SerializableConsumer<NDArray> {

   /**
    * Glorot and Bengio (2010) for sigmoid units
    */
   NDArrayInitializer glorotAndBengioSigmoid = (m) -> {
      double max = 4 * Math.sqrt(6.0) / Math.sqrt(m.rows() + m.columns());
      double min = -max;
      m.mapi(x -> min + (max - min) * Math.random());
   };

   /**
    * Glorot and Bengio (2010) for hyperbolic tangent units
    */
   NDArrayInitializer glorotAndBengioTanH = (m) -> {
      double max = Math.sqrt(6.0) / Math.sqrt(m.rows() + m.columns());
      double min = -max;
      m.mapi(x -> min + (max - min) * Math.random());
   };

   /**
    * Rand nd array initializer.
    *
    * @param rnd the rnd
    * @return the nd array initializer
    */
   static NDArrayInitializer rand(Random rnd) {
      return (m) -> m.mapi(d -> rnd.nextDouble());
   }

   Random rnd = new Random(123);

   /**
    * Rand nd array initializer.
    */
   NDArrayInitializer rand = (m) -> m.mapi(d -> rnd.nextDouble());

   /**
    * Rand nd array initializer.
    *
    * @param rnd the rnd
    * @param min the min
    * @param max the max
    * @return the nd array initializer
    */
   static NDArrayInitializer rand(Random rnd, int min, int max) {
      return (m) -> m.mapi(d -> min + rnd.nextDouble() * max);
   }

   /**
    * Rand nd array initializer.
    *
    * @param min the min
    * @param max the max
    * @return the nd array initializer
    */
   static NDArrayInitializer rand(int min, int max) {
      return rand(new Random(), min, max);
   }

   /**
    * Randn nd array initializer.
    *
    * @param rnd the rnd
    * @return the nd array initializer
    */
   static NDArrayInitializer randn(Random rnd) {
      return (m) -> m.mapi(d -> rnd.nextGaussian());
   }

   /**
    * Randn nd array initializer.
    */
   NDArrayInitializer randn = randn(new Random());

   /**
    * The constant ZEROES.
    */
   NDArrayInitializer zeroes = (m) -> m.mapi(x -> 0d);

   /**
    * The constant Ones.
    */
   NDArrayInitializer ones = (m) -> m.mapi(x -> 1d);


}//END OF NDArrayInitializer
