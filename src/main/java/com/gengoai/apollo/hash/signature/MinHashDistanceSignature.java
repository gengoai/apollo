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

import com.gengoai.Parameters;
import com.gengoai.apollo.hash.LSHParameter;
import com.gengoai.apollo.stat.measure.Measure;
import com.gengoai.apollo.stat.measure.Similarity;

/**
 * <p>Signature function for Jaccard distance / similarity. Uses the Jaccard distance as its measure.</p>
 *
 * @author David B. Bracewell
 */
public class MinHashDistanceSignature extends MinHashSignature {
   public static final String NAME = "MIN_HASH_DISTANCE";
   private static final long serialVersionUID = 1L;


   /**
    * Instantiates a new Min hash signature.
    */
   public MinHashDistanceSignature(Parameters<LSHParameter> parameters) {
      super(parameters);
   }

   @Override
   public Measure getMeasure() {
      return Similarity.Jaccard.asDistanceMeasure();
   }
}// END OF MinHashDistanceSignature
