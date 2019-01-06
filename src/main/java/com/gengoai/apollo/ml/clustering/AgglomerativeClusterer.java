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
import com.gengoai.apollo.ml.FitParameters;
import com.gengoai.apollo.ml.ModelParameters;
import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.apollo.ml.preprocess.Preprocessor;
import com.gengoai.collection.Iterables;
import com.gengoai.conversion.Cast;
import com.gengoai.stream.MStream;
import com.gengoai.tuple.Tuple3;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static com.gengoai.tuple.Tuples.$;

/**
 * <p>
 * Implementation of Hierarchical Agglomerative Clustering
 * </p>
 *
 * @author David B. Bracewell
 */
public class AgglomerativeClusterer extends Clusterer {
   private static final long serialVersionUID = 1L;
   private final AtomicInteger idGenerator = new AtomicInteger();

   /**
    * Instantiates a new Agglomerative clusterer.
    *
    * @param preprocessors the preprocessors
    */
   public AgglomerativeClusterer(Preprocessor... preprocessors) {
      super(ModelParameters.indexedLabelVectorizer()
                           .preprocessors(preprocessors));
   }

   /**
    * Instantiates a new Agglomerative clusterer.
    *
    * @param modelParameters the model parameters
    */
   public AgglomerativeClusterer(ModelParameters modelParameters) {
      super(modelParameters);
   }

   private void doTurn(PriorityQueue<Tuple3<Cluster, Cluster, Double>> priorityQueue,
                       List<Cluster> clusters,
                       Parameters parameters
                      ) {
      Tuple3<Cluster, Cluster, Double> minC = priorityQueue.remove();
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
                                      .map(c2 -> $(cprime, c2,
                                                   parameters.linkage.calculate(cprime, c2, parameters.measure)))
                                      .collect(Collectors.toList()));

         clusters.add(cprime);
      }

   }

   /**
    * Fit clustering.
    *
    * @param stream     the stream
    * @param parameters the parameters
    * @return the clustering
    */
   public Clustering fit(MStream<NDArray> stream, Parameters parameters) {
      idGenerator.set(0);
      List<NDArray> instances = stream.collect();
      PriorityQueue<Tuple3<Cluster, Cluster, Double>> priorityQueue = new PriorityQueue<>(
         Comparator.comparingDouble(Tuple3::getV3));
      List<Cluster> clusters = initDistanceMatrix(instances, priorityQueue, parameters);
      while (clusters.size() > 1) {
         doTurn(priorityQueue, clusters, parameters);
      }
      HierarchicalClustering clustering = new HierarchicalClustering(parameters.measure);
      clustering.root = clusters.get(0);
      return clustering;
   }

   @Override
   protected Clustering fitPreprocessed(Dataset preprocessed, FitParameters fitParameters) {
      return fit(preprocessed.stream().map(this::encode), Cast.as(fitParameters, Parameters.class));
   }

   @Override
   public FitParameters getDefaultFitParameters() {
      return new Parameters();
   }

   private List<Cluster> initDistanceMatrix(List<NDArray> instances,
                                            PriorityQueue<Tuple3<Cluster, Cluster, Double>> priorityQueue,
                                            Parameters parameters
                                           ) {
      List<Cluster> clusters = new ArrayList<>();
      for (NDArray item : instances) {
         Cluster c = new Cluster();
         c.addPoint(item);
         c.setId(idGenerator.getAndIncrement());
         clusters.add(c);
      }


      priorityQueue.addAll(IntStream.range(0, instances.size() - 2)
                                    .boxed()
                                    .flatMap(i -> IntStream.range(i + 1, instances.size())
                                                           .boxed()
                                                           .map(j -> $(clusters.get(i), clusters.get(j))))
                                    .parallel()
                                    .map(
                                       t -> $(t.v1, t.v2, parameters.linkage.calculate(t.v1, t.v2, parameters.measure)))
                                    .collect(Collectors.toList()));
      return clusters;
   }

   /**
    * {@link FitParameters} for Agglomerative Clustering
    */
   public static class Parameters extends ClusterParameters {
      private static final long serialVersionUID = 1L;
      /**
       * The linkage to use for computing the distance between clusters
       */
      public Linkage linkage = Linkage.Complete;
   }

}//END OF AgglomerativeClusterer
