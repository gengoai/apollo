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
import com.davidbracewell.guava.common.util.concurrent.AtomicDouble;
import com.davidbracewell.stream.MStream;
import com.davidbracewell.tuple.Tuple2;
import lombok.Data;
import lombok.Getter;
import lombok.NonNull;
import lombok.Setter;
import lombok.experimental.Accessors;
import org.eclipse.collections.api.set.primitive.MutableIntSet;
import org.eclipse.collections.impl.map.mutable.primitive.IntIntHashMap;
import org.eclipse.collections.impl.set.mutable.primitive.IntHashSet;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.IntStream;

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


   /**
    * Instantiates a new K medoids.
    *
    * @param K the k
    */
   public KMedoids(int K, @NonNull DistanceMeasure distanceMeasure) {
      this.K = K;
      this.distanceMeasure = distanceMeasure;
   }

   @Override
   public FlatClustering cluster(@NonNull MStream<Vector> instanceStream) {
      final List<Vector> instances = instanceStream.collect();
      Map<Tuple2<Integer, Integer>, Double> distanceCache = new ConcurrentHashMap<>();

      List<TempCluster> tempClusters = new ArrayList<>();
      IntHashSet seen = new IntHashSet();
      while (seen.size() < K) {
         seen.add((int) Math.round(Math.random() * K));
      }
      seen.forEach(i -> tempClusters.add(new TempCluster().centroid(i)));

      IntIntHashMap assignments = new IntIntHashMap();

      for (int iteration = 0; iteration < maxIterations; iteration++) {
         System.err.println("iteration " + iteration);
         AtomicLong numChanged = new AtomicLong();
         tempClusters.forEach(c -> c.points().clear());
         IntStream.range(0, instances.size()).parallel()
                  .forEach((int i) -> {
                     double minDistance = Double.POSITIVE_INFINITY;
                     int minC = -1;
                     for (int c = 0; c < tempClusters.size(); c++) {
                        TempCluster cluster = tempClusters.get(c);
                        if (cluster.centroid == i) {
                           minC = c;
                           break;
                        }
                        double d = distance(i, cluster.centroid, instances, distanceCache);
                        if (d < minDistance) {
                           minC = c;
                           minDistance = d;
                        }
                     }
                     int old = assignments.getIfAbsent(i, -1);
                     assignments.put(i, minC);
                     if (old != minC) {
                        numChanged.incrementAndGet();
                        tempClusters.get(minC).points().add(i);
                     }
                  });

         if (numChanged.get() == 0) {
            break;
         }


         tempClusters.parallelStream()
                     .forEach(c -> {
                        AtomicInteger minPoint = new AtomicInteger();
                        AtomicDouble minDistance = new AtomicDouble(Double.POSITIVE_INFINITY);
                        c.points().forEach(i -> {
                           AtomicDouble sum = new AtomicDouble();
                           AtomicLong total = new AtomicLong();
                           c.points().forEach(j -> {
                              total.incrementAndGet();
                              sum.addAndGet(distance(i, j, instances, distanceCache));
                           });
                           double avg = sum.get() / total.get();
                           if (avg < minDistance.get()) {
                              minDistance.set(avg);
                              minPoint.set(i);
                           }
                        });
                        c.centroid(minPoint.get());
                     });
      }

      List<Cluster> finalClusters = new ArrayList<>();
      AtomicInteger cid = new AtomicInteger();
      tempClusters.forEach(tc -> {
         Cluster c = new Cluster();
         c.setId(cid.getAndIncrement());
         c.setCentroid(instances.get(tc.centroid));
         AtomicDouble sum = new AtomicDouble();
         AtomicLong total = new AtomicLong();
         tc.points().forEach(i -> {
            c.addPoint(instances.get(i));
            tc.points().forEach(j -> {
               total.incrementAndGet();
               sum.addAndGet(distance(i, j, instances, distanceCache));
            });
         });
         c.setScore(sum.get() / total.get());
         finalClusters.add(c);
      });


      return new FlatCentroidClustering(getEncoderPair(), getDistanceMeasure(), finalClusters);
   }

   private double distance(int i, int j, List<Vector> instances, Map<Tuple2<Integer, Integer>, Double> distances) {
      return distances.computeIfAbsent($(i, j), t -> distanceMeasure.calculate(instances.get(i), instances.get(j)));
   }

   /**
    * Sets the distance measure to use.
    *
    * @param distanceMeasure the distance measure
    */
   public void setDistanceMeasure(@NonNull DistanceMeasure distanceMeasure) {
      this.distanceMeasure = distanceMeasure;
   }

   @Data
   @Accessors(fluent = true)
   private static class TempCluster {
      int centroid;
      MutableIntSet points = new IntHashSet();
   }

}//END OF KMedoids
