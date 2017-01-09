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
import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.ml.clustering.Cluster;
import com.davidbracewell.apollo.ml.clustering.Clusterer;
import com.davidbracewell.guava.common.base.Throwables;
import com.davidbracewell.guava.common.cache.Cache;
import com.davidbracewell.guava.common.cache.CacheBuilder;
import com.davidbracewell.stream.MStream;
import com.davidbracewell.tuple.Tuple2;
import lombok.Getter;
import lombok.NonNull;
import lombok.Setter;

import java.util.Comparator;
import java.util.List;
import java.util.concurrent.ExecutionException;

import static com.davidbracewell.tuple.Tuples.$;

/**
 * The type K medoids.
 *
 * @author David B. Bracewell
 */
public class KMedoids extends Clusterer<FlatClustering> {
   private static final long serialVersionUID = 1L;
   @Getter
   @Setter
   private int K = 20;
   @Getter
   @Setter
   private int maxIterations = 100;
   @Getter
   private DistanceMeasure distanceMeasure = Distance.Euclidean;

   /**
    * Instantiates a new K medoids.
    */
   public KMedoids() {
   }


   /**
    * Instantiates a new K medoids.
    *
    * @param K the k
    */
   public KMedoids(int K) {
      this.K = K;
   }

   @Override
   public FlatClustering cluster(@NonNull MStream<Vector> instances) {
      Cache<Tuple2<Vector, Vector>, Double> distances = CacheBuilder.newBuilder().build();
      List<Cluster> clusters = instances.sample(false, K)
                                        .map(v -> {
                                           Cluster c = new Cluster();
                                           c.setCentroid(v);
                                           return c;
                                        }).collect();


      for (int iteration = 0; iteration < maxIterations; iteration++) {
         instances.parallel()
                  .forEach(v -> {
                              int minC = -1;
                              double minD = Double.POSITIVE_INFINITY;
                              for (int i = 0; i < clusters.size(); i++) {
                                 Cluster cluster = clusters.get(i);
                                 if (cluster.getCentroid() == v) {
                                    minC = i;
                                    minD = 0;
                                    break;
                                 }
                                 Double dist = null;
                                 try {
                                    dist = distances.get($(v, cluster.getCentroid()),
                                                         () -> distanceMeasure.calculate(v, cluster.getCentroid()));
                                 } catch (ExecutionException e) {
                                    throw Throwables.propagate(e);
                                 }
                                 if (dist < minD) {
                                    minC = i;
                                    minD = 0;
                                 }
                              }
                              clusters.get(minC).addPoint(v);
                           }
                          );

         clusters.forEach(c ->
                             c.setCentroid(c.getPoints()
                                            .parallelStream()
                                            .map(v -> $(v, c.getPoints()
                                                            .stream()
                                                            .mapToDouble(v2 -> {
                                                               Double dist = null;
                                                               try {
                                                                  dist = distances.get($(v, v2),
                                                                                       () -> distanceMeasure.calculate(
                                                                                          v, v2));
                                                               } catch (ExecutionException e) {
                                                                  throw Throwables.propagate(e);
                                                               }
                                                               return dist;
                                                            })
                                                            .sum()))
                                            .min(Comparator.comparingDouble(Tuple2::getValue))
                                            .map(Tuple2::getKey)
                                            .orElse(c.getCentroid()))
                         );
      }

      clusters.forEach(c -> c.setScore(c.getPoints()
                                        .parallelStream()
                                        .flatMapToDouble(v -> c.getPoints()
                                                               .stream()
                                                               .mapToDouble(v2 -> {
                                                                  Double dist = null;
                                                                  try {
                                                                     dist = distances.get($(v, v2),
                                                                                          () -> distanceMeasure.calculate(
                                                                                             v, v2));
                                                                  } catch (ExecutionException e) {
                                                                     throw Throwables.propagate(e);
                                                                  }
                                                                  return dist;
                                                               }))
                                        .average().orElse(0d))
                      );

      return new FlatCentroidClustering(getEncoderPair(), getDistanceMeasure(), clusters);
   }

   /**
    * Sets the distance measure to use.
    *
    * @param distanceMeasure the distance measure
    */
   public void setDistanceMeasure(@NonNull DistanceMeasure distanceMeasure) {
      this.distanceMeasure = distanceMeasure;
   }

}//END OF KMedoids
