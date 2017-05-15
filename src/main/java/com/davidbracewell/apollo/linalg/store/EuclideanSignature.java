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

package com.davidbracewell.apollo.linalg.store;

import com.davidbracewell.apollo.affinity.Distance;
import com.davidbracewell.apollo.affinity.Measure;
import com.davidbracewell.apollo.linalg.SparseVector;
import com.davidbracewell.apollo.linalg.Vector;

/**
 * <p>Signature function for Euclidean distance</p>
 *
 * @author David B. Bracewell
 */
public class EuclideanSignature implements SignatureFunction {
   private static final long serialVersionUID = 1L;
   private final Vector[] randomProjections;
   private final int[] offset;
   private final int[] w;
   private final int dimension;
   private final int signatureSize;

   /**
    * Instantiates a new Euclidean signature.
    *
    * @param signatureSize the signature size controlling the number of random projections
    * @param dimension     the k of the vector
    * @param maxW          the maximum value for the W parameter which controls the random projection
    */
   public EuclideanSignature(int signatureSize, int dimension, int maxW) {
      this.signatureSize = signatureSize;
      this.dimension = dimension;
      this.randomProjections = new Vector[signatureSize];
      this.w = new int[signatureSize];
      this.offset = new int[signatureSize];
      for (int i = 0; i < signatureSize; i++) {
         this.randomProjections[i] = SparseVector.randomGaussian(dimension);
         this.w[i] = (int) Math.round(Math.random() * maxW);
         this.offset[i] = (int) Math.floor(Math.random() * this.w[i]);
      }

   }

   @Override
   public int[] signature(Vector vector) {
      int[] sig = new int[randomProjections.length];
      for (int i = 0; i < signatureSize; i++) {
         sig[i] = (int) Math.round((vector.dot(randomProjections[i]) + offset[i]) / (double) w[i]);
      }
      return sig;
   }

   @Override
   public boolean isBinary() {
      return false;
   }

   @Override
   public int getDimension() {
      return dimension;
   }

   @Override
   public int getSignatureSize() {
      return signatureSize;
   }

   @Override
   public Measure getMeasure() {
      return Distance.Euclidean;
   }
}//END OF EuclideanSignature
