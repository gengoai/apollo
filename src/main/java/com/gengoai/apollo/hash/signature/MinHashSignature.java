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

package com.gengoai.apollo.hash.signature;

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
   private final SignatureParameters parameters;

   /**
    * Instantiates a new Min hash signature.
    */
   public MinHashSignature(SignatureParameters parameters) {
      this.parameters = parameters.copy();
      double error = 1d - parameters.getThreshold(0.0);
      int ss = parameters.getOrDefault(SignatureParameters.SIGNATURE_SIZE,
                                       (int) (1d / (error * error)));
      this.parameters.setSignatureSize(ss);
      Random rnd = new Random();
      this.coefficients = new long[ss][2];
      for (int i = 0; i < ss; i++) {
         this.coefficients[i][0] = rnd.nextInt(parameters.getDimension());
         this.coefficients[i][1] = rnd.nextInt(parameters.getDimension());
      }
   }

   @Override
   public int getDimension() {
      return parameters.getDimension();
   }

   @Override
   public Measure getMeasure() {
      return Similarity.Jaccard;
   }

   @Override
   public int getSignatureSize() {
      return parameters.getSignatureSize();
   }

   private int h(int i, long x) {
      return (int) (coefficients[i][0] * x + coefficients[i][1]) % parameters.getDimension();
   }

   @Override
   public boolean isBinary() {
      return false;
   }

   @Override
   public int[] signature(NDArray vector) {
      int[] sig = new int[parameters.getSignatureSize()];
      Arrays.fill(sig, Integer.MAX_VALUE);
      vector.sparseIterator().forEachRemaining(entry -> {
         if (entry.getValue() > 0) {
            for (int i = 0; i < parameters.getSignatureSize(); i++) {
               sig[i] = Math.min(sig[i], h(i, entry.getIndex()));
            }
         }
      });
      return sig;
   }
}// END OF MinHashSignature
