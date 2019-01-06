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
   public static final String NAME = "COSINE_SIMILARITY";
   private static final long serialVersionUID = 1L;
   private final LSHParameter parameters;
   private final NDArray[] randomProjections;

   /**
    * Instantiates a new Cosine signature.
    */
   public CosineSignature(LSHParameter parameters) {
      this.parameters = parameters.copy();
      this.randomProjections = new NDArray[parameters.signatureSize];
      for (int i = 0; i < randomProjections.length; i++) {
         this.randomProjections[i] = NDArrayFactory.DEFAULT().create(NDArrayInitializer.randn,
                                                                     parameters.dimension);
      }
   }


   @Override
   public Measure getMeasure() {
      return Similarity.Cosine;
   }


   @Override
   public boolean isBinary() {
      return true;
   }

   @Override
   public LSHParameter getParameters() {
      return parameters.copy();
   }

   @Override
   public int[] signature(NDArray vector) {
      int[] sig = new int[randomProjections.length];
      for (int i = 0; i < parameters.signatureSize; i++) {
         sig[i] = randomProjections[i].scalarDot(vector) > 0 ? 1 : 0;
      }
      return sig;
   }

}// END OF CosineSignature
