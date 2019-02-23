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
 *
 */

package com.gengoai.apollo.ml.clustering;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.linear.NDArrayFactory;
import com.gengoai.apollo.ml.DiscretePipeline;
import com.gengoai.apollo.ml.preprocess.Preprocessor;

/**
 * <p>Abstract base class for {@link Clusterer}s whose resultant clustering is flat, i.e. are a set of K lists of
 * points where K is the number of clusters and are defined via centroids, i.e. central points.</p>
 *
 * @author David B. Bracewell
 */
public abstract class FlatCentroidClusterer extends FlatClusterer {
   /**
    * Instantiates a new FlatCentroidClusterer model.
    *
    * @param preprocessors the preprocessors
    */
   public FlatCentroidClusterer(Preprocessor... preprocessors) {
      super(preprocessors);
   }

   /**
    * Instantiates a new FlatCentroidClusterer model.
    *
    * @param modelParameters the model parameters
    */
   public FlatCentroidClusterer(DiscretePipeline modelParameters) {
      super(modelParameters);
   }


   @Override
   public Cluster estimate(NDArray example) {
      return getMeasure().getOptimum().optimum(stream().parallel(),
                                               c -> getMeasure().calculate(example, c.getCentroid()))
                         .orElseThrow(IllegalStateException::new);
   }

   @Override
   public NDArray measure(NDArray example) {
      NDArray distances = NDArrayFactory.DENSE.zeros(size());
      for (int i = 0; i < size(); i++) {
         distances.set(i, getMeasure().calculate(example, get(i).getCentroid()));
      }
      return distances;
   }

}//END OF FlatCentroidClustering
