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

import com.davidbracewell.guava.common.base.Preconditions;

import java.util.Map;

/**
 * <p>Distance measure implementation that is the one minus the value of a similarity measure</p>
 *
 * @author David B. Bracewell
 */
class OneMinusSimilarityDistance implements DistanceMeasure {
   private static final long serialVersionUID = 1L;
   private final SimilarityMeasure similarityMeasure;

   /**
    * Instantiates a new One minus similarity distance.
    *
    * @param similarityMeasure the similarity measure
    */
   public OneMinusSimilarityDistance(SimilarityMeasure similarityMeasure) {
      this.similarityMeasure = Preconditions.checkNotNull(similarityMeasure);
   }

   @Override
   public SimilarityMeasure asSimilarityMeasure() {
      return similarityMeasure;
   }

   @Override
   public double calculate(Map<?, ? extends Number> m1, Map<?, ? extends Number> m2) {
      return 1d - similarityMeasure.calculate(m1, m2);
   }

}//END OF OneMinusSimilarityDistance
