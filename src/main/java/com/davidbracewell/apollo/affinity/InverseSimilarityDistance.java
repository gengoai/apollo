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

package com.davidbracewell.apollo.affinity;

import lombok.NonNull;

import java.util.Map;

/**
 * <p>Distance function that uses the inverse of the result of a similarity function</p>
 *
 * @author David B. Bracewell
 */
class InverseSimilarityDistance implements DistanceMeasure {
   private static final long serialVersionUID = 1L;
   private final SimilarityMeasure similarityMeasure;

   /**
    * Instantiates a new Inverse similarity distance.
    *
    * @param similarityMeasure the similarity measure
    */
   InverseSimilarityDistance(@NonNull SimilarityMeasure similarityMeasure) {
      this.similarityMeasure = similarityMeasure;
   }

   @Override
   public double calculate(Map<?, ? extends Number> m1, Map<?, ? extends Number> m2) {
      return 1d / similarityMeasure.calculate(m1, m2);
   }

   @Override
   public SimilarityMeasure asSimilarityMeasure() {
      return similarityMeasure;
   }

}//END OF OneMinusSimilarityDistance
