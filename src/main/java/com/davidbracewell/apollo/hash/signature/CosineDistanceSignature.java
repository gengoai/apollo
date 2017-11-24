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

package com.davidbracewell.apollo.hash.signature;

import com.davidbracewell.apollo.stat.measure.Measure;
import com.davidbracewell.apollo.stat.measure.Similarity;

/**
 * <p>Signature function for Cosine distance / similarity. Uses the Cosine distance as its measure.</p>
 *
 * @author David B. Bracewell
 */
public class CosineDistanceSignature extends CosineSignature {
   private static final long serialVersionUID = 1L;

   /**
    * Instantiates a new Cosine signature.
    *
    * @param signatureSize the signature size controlling the number of random projections
    * @param dimension     the dimension of the vector
    */
   public CosineDistanceSignature(int signatureSize, int dimension) {
      super(signatureSize, dimension);
   }

   @Override
   public Measure getMeasure() {
      return Similarity.Cosine.asDistanceMeasure();
   }

}// END OF CosineDistanceSignature
