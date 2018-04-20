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

package com.gengoai.apollo.ml.clustering.hierarchical;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.ml.clustering.Cluster;
import com.gengoai.apollo.ml.clustering.Clusterer;
import com.gengoai.apollo.stat.measure.Distance;
import com.gengoai.apollo.stat.measure.DistanceMeasure;
import com.gengoai.guava.common.collect.Iterables;
import com.gengoai.stream.MStream;
import com.gengoai.tuple.Tuple3;
import com.gengoai.apollo.ml.clustering.Cluster;
import com.gengoai.apollo.ml.clustering.Clusterer;
import lombok.Getter;
import lombok.NonNull;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

import static com.gengoai.tuple.Tuples.$;

/**
 * <p>Implementation of Hierarchical Agglomerative Clustering with options for Single, Complete, and Average link
 * clustering.</p>
 *
 * @author David B. Bracewell
 */
public class AgglomerativeClusterer extends Clusterer<HierarchicalClustering> {
   private static final long serialVersionUID = 1L;
   private final AtomicInteger idGenerator = new AtomicInteger();
   @Getter
   private DistanceMeasure distanceMeasure = Distance.Euclidean;
   @Getter
   private Linkage linkage = Linkage.Single;

   /**
    * Instantiates a new Agglomerative clusterer.
    */
   public AgglomerativeClusterer() {

   }

   /**
    * Instantiates a new Agglomerative clusterer.
    *
    * @param distanceMeasure the distance measure to use (default Euclidean)
    * @param linkage         the linkage type to use (default Single link)
    */
   public AgglomerativeClusterer(@NonNull DistanceMeasure distanceMeasure, @NonNull Linkage linkage) {
      this.distanceMeasure = distanceMeasure;
      this.linkage = linkage;
   }

   @Override
   public HierarchicalClustering cluster(@NonNull MStream<NDArray> instanceStream) {
      List<NDArray> instances = instanceStream.collect();
      PriorityQueue<Tuple3<Cluster, Cluster, Double>> priorityQueue = new PriorityQueue<>(Comparator.comparingDouble(
         Tuple3::getV3));

//      LoadingCache<Tuple2<Cluster, Cluster>, Double> distanceCache = CacheBuilder
//                                                                        .newBuilder()
//                                                                        .concurrencyLevel(
//                                                                           SystemInfo.NUMBER_OF_PROCESSORS)
//                                                                        .maximumSize(
//                                                                           instanceStream.count() * instanceStream.count())
//                                                                        .expireAfterWrite(10, TimeUnit.MINUTES)
//                                                                        .build(
//                                                                           new CacheLoader<Tuple2<Cluster, Cluster>, Double>() {
//                                                                              @Override
//                                                                              public Double load(Tuple2<Cluster, Cluster> objects) throws Exception {
//                                                                                 return linkage.calculate(objects.v1,
//                                                                                                          objects.v2,
//                                                                                                          distanceMeasure);
//                                                                              }
//                                                                           });
      List<Cluster> clusters = initDistanceMatrix(instances, priorityQueue);

      while (clusters.size() > 1) {
         doTurn(priorityQueue, clusters);
      }

      HierarchicalClustering clustering = new HierarchicalClustering(this, distanceMeasure);
      clustering.root = clusters.get(0);
      clustering.linkage = linkage;
      return clustering;
   }

   private void doTurn(PriorityQueue<Tuple3<Cluster, Cluster, Double>> priorityQueue, List<Cluster> clusters) {
      Tuple3<Cluster, Cluster, Double> minC = priorityQueue.remove();
      System.err.println(minC);
      if (minC != null) {
         priorityQueue.removeIf(triple -> triple.v2.getId() == minC.v2.getId()
                                             || triple.v1.getId() == minC.v1.getId()
                                             || triple.v2.getId() == minC.v1.getId()
                                             || triple.v1.getId() == minC.v2.getId()
                               );
         Cluster cprime = new Cluster();
         cprime.setId(idGenerator.getAndIncrement());
         cprime.setLeft(minC.getV1());
         cprime.setRight(minC.getV2());
         minC.getV1().setParent(cprime);
         minC.getV2().setParent(cprime);
         cprime.setScore(minC.v3);
         for (NDArray point : Iterables.concat(minC.getV1().getPoints(), minC.getV2().getPoints())) {
            cprime.addPoint(point);
         }
         clusters.remove(minC.getV1());
         clusters.remove(minC.getV2());

         priorityQueue.addAll(clusters.parallelStream()
                                      .map(c2 -> $(cprime, c2, linkage.calculate(cprime, c2, distanceMeasure)))
                                      .collect(Collectors.toList()));

         clusters.add(cprime);
      }

   }

   private List<Cluster> initDistanceMatrix(List<NDArray> instances, PriorityQueue<Tuple3<Cluster, Cluster, Double>> priorityQueue) {
      List<Cluster> clusters = new ArrayList<>();
      for (NDArray item : instances) {
         Cluster c = new Cluster();
         c.addPoint(item);
         c.setId(idGenerator.getAndIncrement());
         clusters.add(c);
      }

      priorityQueue.addAll(clusters.parallelStream()
                                   .flatMap(c1 -> clusters.stream()
                                                          .filter(c2 -> c2.getId() != c1.getId())
                                                          .map(c2 -> $(c1, c2)))
                                   .map(entry -> $(entry.getKey(), entry.getValue(), linkage.calculate(entry.getKey(),
                                                                                                       entry.getValue(),
                                                                                                       distanceMeasure)))
                                   .collect(Collectors.toList()));

      System.err.println(priorityQueue.size());

      return clusters;
   }

//   private void doTurn(LoadingCache<Tuple2<Cluster, Cluster>, Double> distanceMatrix, List<Cluster> clusters) {
//
//      Tuple3<Cluster, Cluster, Double> minC = clusters
//                                                 .parallelStream()
//                                                 .map(c -> clusters
//                                                              .stream()
//                                                              .filter(c2 -> c.getId() != c2.getId())
//                                                              .map(c2 -> {
//                                                                 Double d;
//                                                                 try {
//                                                                    d = distanceMatrix.get($(c, c2));
//                                                                 } catch (ExecutionException e) {
//                                                                    d = Double.POSITIVE_INFINITY;
//                                                                 }
//                                                                 return $(c2, d);
//                                                              })
//                                                              .min(Map.Entry.comparingByValue())
//                                                              .orElseThrow(NullArgumentException::new)
//                                                              .appendLeft(c)
//                                                     )
//                                                 .min(Comparator.comparingDouble(t -> t.v3))
//                                                 .orElseThrow(NullArgumentException::new);
//      if (minC != null) {
//         Cluster cprime = new Cluster();
//         cprime.setId(idGenerator.getAndIncrement());
//         cprime.setLeft(minC.getV1());
//         cprime.setRight(minC.getV2());
//         minC.getV1().setParent(cprime);
//         minC.getV2().setParent(cprime);
//         cprime.setScore(minC.v3);
//
//         clusters.remove(minC.getV1());
//         clusters.remove(minC.getV2());
//
//         for (Vector point : Iterables.concat(minC.getV1().getPoints(), minC.getV2().getPoints())) {
//            cprime.addPoint(point);
//         }
//
//         clusters.add(cprime);
//      }
//
//   }

   @Override
   public void resetLearnerParameters() {
      super.resetLearnerParameters();
      idGenerator.set(0);
   }

   /**
    * Sets the distance measure to use.
    *
    * @param distanceMeasure the distance measure
    */
   public void setDistanceMeasure(@NonNull DistanceMeasure distanceMeasure) {
      this.distanceMeasure = distanceMeasure;
   }

   /**
    * Sets the linkage to use.
    *
    * @param linkage the linkage
    */
   public void setLinkage(@NonNull Linkage linkage) {
      this.linkage = linkage;
   }
}//END OF AgglomerativeClusterer
