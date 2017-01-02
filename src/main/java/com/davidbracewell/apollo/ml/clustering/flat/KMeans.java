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


import com.davidbracewell.apollo.affinity.Distance;
import com.davidbracewell.apollo.affinity.DistanceMeasure;
import com.davidbracewell.apollo.linalg.SparseVector;
import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.ml.clustering.Cluster;
import com.davidbracewell.apollo.ml.clustering.Clusterer;
import com.davidbracewell.guava.common.base.Preconditions;
import com.davidbracewell.stream.MStream;
import lombok.Getter;
import lombok.NonNull;
import lombok.Setter;

import java.util.List;
import java.util.Map;
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
   public FlatCentroidClustering cluster(@NonNull MStream<Vector> instanceStream) {
      FlatCentroidClustering clustering = new FlatCentroidClustering(getEncoderPair(), distanceMeasure);

      List<Vector> instances = instanceStream.collect();
      for (Vector centroid : initCentroids(instances)) {
         Cluster c = new Cluster();
         c.setId(c.size());
         c.setCentroid(centroid);
         clustering.addCluster(c);
      }

      Map<Vector, Integer> assignment = new ConcurrentHashMap<>();

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
            Vector c = clustering.get(i).getCentroid().zero();
            for (Vector ii : clustering.get(i)) {
               if (ii != null) {
                  c.addSelf(ii);
               }
            }
            c.mapDivideSelf((double) clustering.get(i).size());
            double average = 0;
            for (Vector ii : clustering.get(i)) {
               if (ii != null) {
                  average += distanceMeasure.calculate(ii, c);
               }
            }
            clustering.get(i).setId(i);
            clustering.get(i).setScore(average / clustering.get(i).size());
         }
      }

      return clustering;
   }


   /**
    * Sets the distance measure to use.
    *
    * @param distanceMeasure the distance measure
    */
   public void setDistanceMeasure(@NonNull DistanceMeasure distanceMeasure) {
      this.distanceMeasure = distanceMeasure;
   }

   private Vector[] initCentroids(List<Vector> instances) {
      Vector[] centroids = new Vector[K];
      for (int i = 0; i < K; i++) {
         centroids[i] = new SparseVector(instances.get(0).dimension());
      }
      double[] cnts = new double[K];
      Random rnd = new Random();
      for (Vector ii : instances) {
         int ci = rnd.nextInt(K);
         centroids[ci].addSelf(ii);
         cnts[ci]++;
      }
      for (int i = 0; i < K; i++) {
         centroids[i].mapDivideSelf(cnts[i]);
      }
      return centroids;
   }


}//END OF KMeans
