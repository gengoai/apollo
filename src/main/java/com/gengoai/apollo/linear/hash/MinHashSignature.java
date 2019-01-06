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

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.stat.measure.Measure;
import com.gengoai.apollo.stat.measure.Similarity;

import java.util.Arrays;
import java.util.Random;

/**
 * <p>Signature function for Jaccard distance / similarity. Uses the Jaccard similarity as its measure.</p>
 *
 * @author David B. Bracewell
 */
public class MinHashSignature implements SignatureFunction {
   public static final String NAME = "MIN_HASH";
   private static final long serialVersionUID = 1L;
   private final long[][] coefficients;
   private final LSHParameter parameters;

   /**
    * Instantiates a new Min hash signature.
    */
   public MinHashSignature(LSHParameter parameters) {
      this.parameters = parameters.copy();
      double error = 1d - parameters.threshold;
      this.parameters.setIf("signatureSize", (Integer ss) -> ss <= 0, (int) (1d / (error * error)));
      Random rnd = new Random();
      this.coefficients = new long[this.parameters.signatureSize][2];
      for (int i = 0; i < this.parameters.signatureSize; i++) {
         this.coefficients[i][0] = rnd.nextInt(parameters.dimension);
         this.coefficients[i][1] = rnd.nextInt(parameters.dimension);
      }
   }


   @Override
   public Measure getMeasure() {
      return Similarity.Jaccard;
   }


   private int h(int i, long x) {
      return (int) (coefficients[i][0] * x + coefficients[i][1]) % parameters.dimension;
   }

   @Override
   public boolean isBinary() {
      return false;
   }

   @Override
   public LSHParameter getParameters() {
      return parameters.copy();
   }

   @Override
   public int[] signature(NDArray vector) {
      int[] sig = new int[this.parameters.signatureSize];
      Arrays.fill(sig, Integer.MAX_VALUE);
      vector.sparseIterator().forEachRemaining(entry -> {
         if (entry.getValue() > 0) {
            for (int i = 0; i < sig.length; i++) {
               sig[i] = Math.min(sig[i], h(i, entry.getIndex()));
            }
         }
      });
      return sig;
   }
}// END OF MinHashSignature
