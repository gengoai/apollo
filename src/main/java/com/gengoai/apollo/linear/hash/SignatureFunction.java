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

import java.io.Serializable;

/**
 * <p>A signature function converts a high dimensional vector into a low dimensional signature vector that allows for
 * quick nearest neighbor matches.</p>
 *
 * @author David B. Bracewell
 */
public interface SignatureFunction extends Serializable {


   static SignatureFunction create(String name, LSHParameter parameters) {
      switch (name.toUpperCase()) {
         case "JACCARD":
         case "JACCARDSIMILARITY":
         case "MINHASH":
         case "JACCARD_SIMILARITY":
         case "MIN_HASH":
            return new MinHashSignature(parameters);
         case "JACCARDDISTANCE":
         case "JACCARD_DISTANCE":
         case "MINHASHDISTANCE":
         case "MIN_HASH_DISTANCE":
            return new MinHashDistanceSignature(parameters);
         case "COSINE":
         case "COSINESIMILARITY":
         case "COSINE_SIMILARITY":
            return new CosineSignature(parameters);
         case "COSINEDISTANCE":
         case "COSINE_DISTANCE":
            return new CosineDistanceSignature(parameters);
         case "EUCLIDEAN":
         case "EUCLIDEANDISTANCE":
         case "EUCLIDEAN_DISTANCE":
            return new EuclideanSignature(parameters);
      }
      throw new IllegalArgumentException("Unknown value: " + name);
   }

   /**
    * Converts the given vector into a signature
    *
    * @param vector the vector
    * @return the signature
    */
   int[] signature(NDArray vector);

   /**
    * Does this signature function produce real or binary based signatures.
    *
    * @return True if creates binary (boolean) based signatures, False otherwise
    */
   boolean isBinary();

   LSHParameter getParameters();

   /**
    * Gets the measure that is associated with this signature function.
    *
    * @return the measure
    */
   Measure getMeasure();



}//END OF HashingFunction
