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
import com.gengoai.apollo.ml.FitParameters;
import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.apollo.stat.measure.Distance;
import com.gengoai.conversion.Cast;
import org.apache.commons.math3.distribution.MultivariateNormalDistribution;
import org.apache.commons.math3.distribution.fitting.MultivariateNormalMixtureExpectationMaximization;
import org.apache.commons.math3.util.Pair;

import java.util.List;
import java.util.stream.Collectors;

import static com.gengoai.Validation.notNull;

/**
 * @author David B. Bracewell
 */
public class GaussianMixtureModel extends Clusterer {


   @Override
   protected Clustering fitPreprocessed(Dataset preprocessed, FitParameters fitParameters) {
      Parameters p = notNull(Cast.as(fitParameters, Parameters.class));

      List<NDArray> vectors = preprocessed.asVectorStream(getPipeline()).collect();
      int numberOfFeatures = getNumberOfFeatures();
      int numberOfDataPoints = vectors.size();
      double[][] data = new double[numberOfDataPoints][numberOfFeatures];
      for (int i = 0; i < numberOfDataPoints; i++) {
         data[i] = vectors.get(i).toDoubleArray();
      }

      List<MultivariateNormalDistribution> components = MultivariateNormalMixtureExpectationMaximization.estimate(data,
                                                                                                                  p.K)
                                                                                                        .getComponents()
                                                                                                        .stream()
                                                                                                        .map(
                                                                                                           Pair::getSecond)
                                                                                                        .collect(
                                                                                                           Collectors.toList());

      FlatClustering clustering = new FlatClustering(Distance.Euclidean);
      for (int i = 0; i < components.size(); i++) {
         Cluster cluster = new Cluster();
         cluster.setId(i);
         cluster.setCentroid(NDArrayFactory.columnVector(components.get(i).sample()));
      }

      return clustering;
   }

   @Override
   public Parameters getDefaultFitParameters() {
      return new Parameters();
   }

   public static class Parameters extends FitParameters {
      public int K = 100;
   }

}//END OF GaussianMixtureModel
