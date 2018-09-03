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
import com.gengoai.apollo.linear.NDArrayFactory;
import com.gengoai.apollo.linear.NDArrayInitializer;
import com.gengoai.apollo.stat.measure.Measure;
import com.gengoai.apollo.stat.measure.Similarity;

/**
 * <p>Signature function for Cosine distance / similarity. Uses the Cosine similarity as its measure.</p>
 *
 * @author David B. Bracewell
 */
public class CosineSignature implements SignatureFunction {
   private static final long serialVersionUID = 1L;

   private final int dimension;
   private final int signatureSize;
   private final NDArray[] randomProjections;

   /**
    * Instantiates a new Cosine signature.
    *
    * @param signatureSize the signature size controlling the number of random projections
    * @param dimension     the dimension of the vector
    */
   public CosineSignature(int signatureSize, int dimension) {
      this.signatureSize = signatureSize;
      this.dimension = dimension;
      this.randomProjections = new NDArray[signatureSize];
      for (int i = 0; i < signatureSize; i++) {
         this.randomProjections[i] = NDArrayFactory.DEFAULT().create(NDArrayInitializer.randn, dimension);
      }
   }

   @Override
   public int getDimension() {
      return dimension;
   }

   @Override
   public Measure getMeasure() {
      return Similarity.Cosine;
   }

   @Override
   public int getSignatureSize() {
      return signatureSize;
   }

   @Override
   public boolean isBinary() {
      return true;
   }

   @Override
   public int[] signature(NDArray vector) {
      int[] sig = new int[randomProjections.length];
      for (int i = 0; i < signatureSize; i++) {
         sig[i] = randomProjections[i].scalarDot(vector) > 0 ? 1 : 0;
      }
      return sig;
   }

}// END OF CosineSignature
