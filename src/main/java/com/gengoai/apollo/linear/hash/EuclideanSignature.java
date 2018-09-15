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
 */

package com.gengoai.apollo.linear.hash;

import com.gengoai.Parameters;
import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.linear.NDArrayFactory;
import com.gengoai.apollo.linear.NDArrayInitializer;
import com.gengoai.apollo.stat.measure.Distance;
import com.gengoai.apollo.stat.measure.Measure;

import static com.gengoai.apollo.linear.hash.LSHParameter.*;

/**
 * <p>Signature function for Euclidean distance</p>
 *
 * @author David B. Bracewell
 */
public class EuclideanSignature implements SignatureFunction {
   public static final String NAME = "EUCLIDEAN_DISTANCE";
   private static final long serialVersionUID = 1L;
   private final NDArray[] randomProjections;
   private final int[] offset;
   private final int[] w;
   private final Parameters<LSHParameter> parameters;

   /**
    * Instantiates a new Euclidean signature.
    */
   public EuclideanSignature(Parameters<LSHParameter> parameters) {
      this.parameters = parameters.copy();
      this.randomProjections = new NDArray[parameters.<Integer>get(SIGNATURE_SIZE)];
      this.w = new int[parameters.<Integer>get(SIGNATURE_SIZE)];
      this.offset = new int[parameters.<Integer>get(SIGNATURE_SIZE)];
      for (int i = 0; i < parameters.<Integer>get(SIGNATURE_SIZE); i++) {
         this.randomProjections[i] = NDArrayFactory.DEFAULT().create(NDArrayInitializer.randn,
                                                                     parameters.getInt(DIMENSION));
         this.w[i] = (int) Math.round(Math.random() * parameters.getInt(MAX_W));
         this.offset[i] = (int) Math.floor(Math.random() * this.w[i]);
      }

   }

   @Override
   public Measure getMeasure() {
      return Distance.Euclidean;
   }

   @Override
   public Parameters<LSHParameter> getParameters() {
      return parameters;
   }

   @Override
   public boolean isBinary() {
      return false;
   }

   @Override
   public int[] signature(NDArray vector) {
      int[] sig = new int[randomProjections.length];
      for (int i = 0; i < parameters.<Integer>get(SIGNATURE_SIZE); i++) {
         sig[i] = (int) Math.round((vector.scalarDot(randomProjections[i]) + offset[i]) / (double) w[i]);
         sig[i] = sig[i] >= 1 ? 1 : 0;
      }
      return sig;
   }

}//END OF EuclideanSignature
