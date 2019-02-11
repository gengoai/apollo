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

import com.gengoai.Stopwatch;
import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.linear.NDArrayFactory;
import com.gengoai.apollo.linear.NDArrayInitializer;
import com.gengoai.apollo.ml.DiscretePipeline;
import com.gengoai.apollo.ml.FitParameters;
import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.apollo.ml.preprocess.Preprocessor;
import com.gengoai.apollo.stat.measure.Measure;
import com.gengoai.conversion.Cast;
import com.gengoai.function.SerializableSupplier;
import com.gengoai.logging.Logger;
import com.gengoai.stream.MStream;

import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Random;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.IntStream;

/**
 * <p>
 * Implementation of <a href="https://en.wikipedia.org/wiki/K-means_clustering">K-means</a> Clustering using Loyd's
 * algorithm.
 * </p>
 *
 * @author David B. Bracewell
 */
public class KMeans extends Clusterer {
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


   /**
    * Clusters the given points using the given K-means fit parameters
    *
    * @param dataSupplier  the data supplier
    * @param fitParameters the fit parameters
    * @return the flat clustering
    */
   public FlatClustering fit(SerializableSupplier<MStream<NDArray>> dataSupplier, Parameters fitParameters) {
      FlatClustering clusters = new FlatClustering(fitParameters.measure);
      List<NDArray> instances = dataSupplier.get().collect();
      for (NDArray centroid : initCentroids(fitParameters.K, instances)) {
         Cluster c = new Cluster();
         c.setCentroid(centroid);
         clusters.add(c);
      }


      final Measure measure = fitParameters.measure;
      Map<NDArray, Integer> assignment = new ConcurrentHashMap<>();

      final AtomicLong numMoved = new AtomicLong(0);
      double lastVariance = 0;

      for (int itr = 0; itr < fitParameters.maxIterations; itr++) {
         Stopwatch sw = Stopwatch.createStarted();
         clusters.forEach(Cluster::clear);
         numMoved.set(0);
         instances.parallelStream()
                  .forEach(ii -> {
                              int minI = 0;
                              double minD = measure.calculate(ii, clusters.get(0).getCentroid());
                              for (int ci = 1; ci < fitParameters.K; ci++) {
                                 double distance = measure.calculate(ii, clusters.get(ci).getCentroid());
                                 if (distance < minD) {
                                    minD = distance;
                                    minI = ci;
                                 }
                              }
                              Integer old = assignment.put(ii, minI);
                              clusters.get(minI).addPoint(ii);
                              if (old == null || old != minI) {
                                 numMoved.incrementAndGet();
                              }
                           }
                          );

         for (int i = 0; i < fitParameters.K; i++) {
            clusters.get(i).getPoints().removeIf(Objects::isNull);
            if (clusters.get(i).size() == 0) {
               clusters.get(i).setCentroid(
                  NDArrayFactory.DEFAULT().create(NDArrayInitializer.rand(-1, 1), (int) instances.get(0).length()));
            } else {
               NDArray c = clusters.get(i).getCentroid().zero();
               for (NDArray ii : clusters.get(i)) {
                  if (ii != null) {
                     c.addi(ii);
                  }
               }
               c.divi((float) clusters.get(i).size());
            }
         }

         sw.stop();
         double variance = clusters.inGroupVariance();
         if (fitParameters.verbose) {
            log.info("iteration={0}: number_moved={1}, variance={2} ({3})", (itr + 1), numMoved, variance, sw);
         }

         if (numMoved.get() == 0 || (itr > 0 && Math.abs(variance - lastVariance) <= fitParameters.tolerance)) {
            break;
         }
         lastVariance = variance;
      }

      for (int i = 0; i < clusters.size(); i++) {
         Cluster cluster = clusters.get(i);
         cluster.setId(i);
         if (cluster.size() > 0) {
            cluster.getPoints().removeIf(Objects::isNull);
            double average = cluster.getPoints().parallelStream()
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
      return clusters;
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


   @Override
   public Clustering fitPreprocessed(Dataset dataSupplier, FitParameters fitParameters) {
      return fit(() -> dataSupplier.asVectorStream(getPipeline()), Cast.as(fitParameters, Parameters.class));
   }


   @Override
   public Parameters getDefaultFitParameters() {
      return new Parameters();
   }


   /**
    * Fit Parameters for KMeans
    */
   public static class Parameters extends ClusterParameters {
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
