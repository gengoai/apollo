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
import com.gengoai.apollo.linear.NDArrayInitializer;
import com.gengoai.apollo.ml.DiscretePipeline;
import com.gengoai.apollo.ml.FitParameters;
import com.gengoai.apollo.ml.preprocess.Preprocessor;
import com.gengoai.apollo.optimization.StoppingCriteria;
import com.gengoai.apollo.statistics.measure.Measure;
import com.gengoai.conversion.Cast;
import com.gengoai.logging.Logger;
import com.gengoai.stream.MStream;

import java.util.List;
import java.util.Objects;
import java.util.Random;
import java.util.stream.IntStream;

/**
 * <p>
 * Implementation of <a href="https://en.wikipedia.org/wiki/K-means_clustering">K-means</a> Clustering using Loyd's
 * algorithm.
 * </p>
 *
 * @author David B. Bracewell
 */
public class KMeans extends FlatCentroidClusterer {
   private static final Logger log = Logger.getLogger(KMeans.class);
   private static final long serialVersionUID = 1L;

   /**
    * Instantiates a new K-Means model.
    *
    * @param preprocessors the preprocessors
    */
   public KMeans(Preprocessor... preprocessors) {
      super(preprocessors);
   }

   /**
    * Instantiates a new K-Means model.
    *
    * @param modelParameters the model parameters
    */
   public KMeans(DiscretePipeline modelParameters) {
      super(modelParameters);
   }

   @Override
   public void fit(MStream<NDArray> vectors, FitParameters parameters) {
      Parameters fitParameters = Cast.as(parameters);
      setMeasure(fitParameters.measure);

      List<NDArray> instances = vectors.collect();
      for (NDArray centroid : initCentroids(fitParameters.K, instances)) {
         Cluster c = new Cluster();
         c.setCentroid(centroid);
         add(c);
      }
      final Measure measure = fitParameters.measure;

      StoppingCriteria.create("numPointsChanged")
                      .historySize(3)
                      .maxIterations(fitParameters.maxIterations)
                      .tolerance(fitParameters.tolerance)
                      .reportInterval(1)
                      .logger(log)
                      .untilTermination(itr -> this.iteration(instances));

      for (int i = 0; i < size(); i++) {
         Cluster cluster = get(i);
         cluster.setId(i);
         if (cluster.size() > 0) {
            cluster.getPoints().removeIf(Objects::isNull);
            double average = cluster.getPoints()
                                    .parallelStream()
                                    .flatMapToDouble(p1 -> cluster.getPoints()
                                                                  .stream()
                                                                  .filter(p2 -> p2 != p1)
                                                                  .mapToDouble(p2 -> measure.calculate(p1, p2)))
                                    .summaryStatistics()
                                    .getAverage();
            cluster.setScore(average);
         } else {
            cluster.setScore(Double.MAX_VALUE);
         }
      }
   }

   @Override
   public Parameters getDefaultFitParameters() {
      return new Parameters();
   }

   private NDArray[] initCentroids(int K, List<NDArray> instances) {
      NDArray[] clusters = IntStream.range(0, K)
                                    .mapToObj(i -> NDArrayFactory.DEFAULT().zeros((int) instances.get(0).length()))
                                    .toArray(NDArray[]::new);

      double[] cnts = new double[K];
      Random rnd = new Random();
      for (NDArray ii : instances) {
         int ci = rnd.nextInt(K);
         clusters[ci].addi(ii);
         cnts[ci]++;
      }
      for (int i = 0; i < K; i++) {
         if (cnts[i] > 0) {
            clusters[i].divi((float) cnts[i]);
         }
      }
      return clusters;
   }

   private double iteration(List<NDArray> instances) {
      //Clear the points
      keepOnlyCentroids();

      final Object[] locks = new Object[size()];
      for (int i = 0; i < size(); i++) {
         locks[i] = new Object();
      }
      //Assign points
      instances.parallelStream().forEach(v -> {
         Cluster c = estimate(v);
         synchronized (locks[c.getId()]) {
            c.addPoint(v);
         }
      });


      double numChanged = 0;
      for (Cluster cluster : this) {
         NDArray centroid;

         //Calculate the new centroid, randomly generating a new vector when the custer has 0 members
         if (cluster.size() == 0) {
            centroid = NDArrayFactory.DEFAULT().create(NDArrayInitializer.rand(-1, 1), (int) instances.get(0).length());
         } else {
            centroid = NDArrayFactory.DENSE.zeros(getNumberOfFeatures());
            for (NDArray point : cluster.getPoints()) {
               centroid.addi(point);
            }
            centroid.divi(cluster.size());
         }
         cluster.setCentroid(centroid);

         //Calculate the number of points tht changed from the previous iteration
         numChanged += cluster.getPoints()
                              .parallelStream()
                              .mapToDouble(n -> {
                                 if (n.getPredicted() == null) {
                                    n.setPredicted((double) cluster.getId());
                                    return 1.0;
                                 }
                                 double c = n.getPredictedAsDouble() == cluster.getId() ? 0 : 1;
                                 n.setPredicted((double) cluster.getId());
                                 return c;
                              }).sum();
      }

      return numChanged;
   }

   /**
    * Fit Parameters for KMeans
    */
   public static class Parameters extends ClusterParameters<Parameters> {
      /**
       * The number of clusters
       */
      public int K = 2;
      /**
       * The maximum number of iterations to run the clusterer for
       */
      public int maxIterations = 100;
      /**
       * The tolerance in change of in-group variance for determining if k-means has converged
       */
      public double tolerance = 1e-3;

   }
}//END OF KMeans
