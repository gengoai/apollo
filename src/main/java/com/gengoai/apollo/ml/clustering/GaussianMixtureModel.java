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
import com.gengoai.apollo.ml.FitParameters;
import com.gengoai.apollo.ml.Params;
import com.gengoai.apollo.ml.preprocess.Preprocessor;
import com.gengoai.conversion.Cast;
import com.gengoai.stream.MStream;
import org.apache.commons.math3.distribution.MultivariateNormalDistribution;
import org.apache.commons.math3.distribution.fitting.MultivariateNormalMixtureExpectationMaximization;
import org.apache.commons.math3.util.Pair;

import java.util.List;
import java.util.stream.Collectors;

/**
 * <p>Gaussian Mixture Model</p>
 *
 * @author David B. Bracewell
 */
public class GaussianMixtureModel extends FlatCentroidClusterer {
   private static final long serialVersionUID = 1L;

   /**
    * Instantiates a new Gaussian mixture model.
    *
    * @param preprocessors the preprocessors
    */
   public GaussianMixtureModel(Preprocessor... preprocessors) {
      super(preprocessors);
   }

   /**
    * Instantiates a new Gaussian mixture model.
    *
    * @param modelParameters the model parameters
    */
   public GaussianMixtureModel(DiscretePipeline modelParameters) {
      super(modelParameters);
   }


   @Override
   public void fit(MStream<NDArray> vectors, FitParameters fp) {
      Parameters p = Cast.as(fp);
      List<NDArray> vectorList = vectors.collect();
      int numberOfFeatures = getNumberOfFeatures();
      int numberOfDataPoints = vectorList.size();
      double[][] data = new double[numberOfDataPoints][numberOfFeatures];
      for (int i = 0; i < numberOfDataPoints; i++) {
         data[i] = vectorList.get(i).toDoubleArray();
      }

      List<MultivariateNormalDistribution> components =
         MultivariateNormalMixtureExpectationMaximization.estimate(data, p.K.value())
                                                         .getComponents()
                                                         .stream()
                                                         .map(Pair::getSecond)
                                                         .collect(Collectors.toList());

      for (int i = 0; i < components.size(); i++) {
         Cluster cluster = new Cluster();
         cluster.setId(i);
         cluster.setCentroid(NDArrayFactory.ND.columnVector(components.get(i).sample()));
         add(cluster);
      }

   }

   @Override
   public Parameters getFitParameters() {
      return new Parameters();
   }

   /**
    * The type Parameters.
    */
   public static class Parameters extends ClusterParameters<Parameters> {
      /**
       * The K.
       */
      public final Parameter<Integer> K = parameter(Params.Clustering.K, 100);
   }

}//END OF GaussianMixtureModel
