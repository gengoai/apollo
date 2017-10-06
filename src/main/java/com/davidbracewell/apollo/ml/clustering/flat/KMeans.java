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

package com.davidbracewell.apollo.ml.clustering.flat;


import com.davidbracewell.apollo.linear.NDArray;
import com.davidbracewell.apollo.linear.NDArrayFactory;
import com.davidbracewell.apollo.ml.clustering.Cluster;
import com.davidbracewell.apollo.ml.clustering.Clusterer;
import com.davidbracewell.apollo.stat.measure.Distance;
import com.davidbracewell.apollo.stat.measure.DistanceMeasure;
import com.davidbracewell.guava.common.base.Preconditions;
import com.davidbracewell.stream.MStream;
import lombok.Getter;
import lombok.NonNull;
import lombok.Setter;

import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Random;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * The type K means.
 *
 * @author David B. Bracewell
 */
public class KMeans extends Clusterer<FlatClustering> {
   private static final long serialVersionUID = 1L;
   @Getter
   @Setter
   private int K;
   @Getter
   @Setter
   private int maxIterations;
   @Getter
   private DistanceMeasure distanceMeasure;

   private KMeans() {
      this(1, Distance.Euclidean, 20);
   }

   /**
    * Instantiates a new K-Means clusterer
    *
    * @param k               the number of clusters
    * @param distanceMeasure the distance measure to use
    */
   public KMeans(int k, DistanceMeasure distanceMeasure) {
      this(k, distanceMeasure, 20);
   }

   /**
    * Instantiates a new K-Means clusterer
    *
    * @param k               the number of clusters
    * @param distanceMeasure the distance measure to use (default Euclidean)
    * @param maxIterations   the maximum number of iterations to run the algorithm (default 20)
    */
   public KMeans(int k, DistanceMeasure distanceMeasure, int maxIterations) {
      Preconditions.checkArgument(k > 0);
      Preconditions.checkArgument(maxIterations > 0);
      this.K = k;
      this.maxIterations = maxIterations;
      this.distanceMeasure = Preconditions.checkNotNull(distanceMeasure);
   }

   @Override
   public FlatCentroidClustering cluster(@NonNull MStream<NDArray> instanceStream) {
      FlatCentroidClustering clustering = new FlatCentroidClustering(this, distanceMeasure);

      List<NDArray> instances = instanceStream.collect();
      for (NDArray centroid : initCentroids(instances)) {
         Cluster c = new Cluster();
         c.setId(c.size());
         c.setCentroid(centroid);
         clustering.addCluster(c);
      }

      Map<NDArray, Integer> assignment = new ConcurrentHashMap<>();

      final AtomicLong numMoved = new AtomicLong(0);
      for (int itr = 0; itr < maxIterations; itr++) {
         clustering.forEach(Cluster::clear);
         numMoved.set(0);
         instances.parallelStream()
                  .forEach(ii -> {
                              int minI = 0;
                              double minD = distanceMeasure.calculate(ii, clustering.get(0).getCentroid());
                              for (int ci = 1; ci < K; ci++) {
                                 double distance = distanceMeasure.calculate(ii, clustering.get(ci).getCentroid());
                                 if (distance < minD) {
                                    minD = distance;
                                    minI = ci;
                                 }
                              }
                              Integer old = assignment.put(ii, minI);
                              clustering.get(minI).addPoint(ii);
                              if (old == null || old != minI) {
                                 numMoved.incrementAndGet();
                              }
                           }
                          );


         if (numMoved.get() == 0) {
            break;
         }

         for (int i = 0; i < K; i++) {
            clustering.get(i).getPoints().removeIf(Objects::isNull);
            if (clustering.get(i).size() == 0) {
               clustering.get(i).setCentroid(NDArrayFactory.defaultFactory().rand(instances.get(0).length(), -1, 1));
            } else {
               NDArray c = clustering.get(i).getCentroid().zero();
               for (NDArray ii : clustering.get(i)) {
                  if (ii != null) {
                     c.addi(ii);
                  }
               }
               c.divi((double) clustering.get(i).size());
            }
         }
      }

      for (int i = 0; i < clustering.size(); i++) {
         Cluster cluster = clustering.get(i);
         cluster.setId(i);
         if (cluster.size() > 0) {
            cluster.getPoints().removeIf(Objects::isNull);
            double average = cluster.getPoints().parallelStream()
                                    .flatMapToDouble(p1 -> cluster.getPoints()
                                                                  .stream()
                                                                  .filter(p2 -> p2 != p1)
                                                                  .mapToDouble(p2 -> distanceMeasure.calculate(p1, p2)))
                                    .summaryStatistics()
                                    .getAverage();
            cluster.setScore(average);
         } else {
            cluster.setScore(Double.MAX_VALUE);
         }
      }

      return clustering;
   }

   private NDArray[] initCentroids(List<NDArray> instances) {
      NDArray[] centroids = new NDArray[K];
      for (int i = 0; i < K; i++) {
         centroids[i] = NDArrayFactory.defaultFactory().zeros(instances.get(0).length());
      }
      double[] cnts = new double[K];
      Random rnd = new Random();
      for (NDArray ii : instances) {
         int ci = rnd.nextInt(K);
         centroids[ci].addi(ii);
         cnts[ci]++;
      }
      for (int i = 0; i < K; i++) {
         centroids[i].divi(cnts[i]);
      }
      return centroids;
   }

   /**
    * Sets the distance measure to use.
    *
    * @param distanceMeasure the distance measure
    */
   public void setDistanceMeasure(@NonNull DistanceMeasure distanceMeasure) {
      this.distanceMeasure = distanceMeasure;
   }


}//END OF KMeans
