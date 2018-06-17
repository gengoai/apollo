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

package com.gengoai.apollo.ml.clustering.flat;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.ml.clustering.Cluster;
import com.gengoai.apollo.ml.clustering.Clusterer;
import com.gengoai.apollo.stat.measure.Distance;
import com.gengoai.apollo.stat.measure.DistanceMeasure;
import com.gengoai.logging.Loggable;
import com.gengoai.stream.MStream;
import com.gengoai.stream.StreamingContext;
import lombok.Getter;
import lombok.NonNull;
import lombok.Setter;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * @author David B. Bracewell
 */
public class DivisiveKMeans extends Clusterer<FlatClustering> implements Loggable {
   private static final long serialVersionUID = 1L;
   @Getter
   private DistanceMeasure distanceMeasure = Distance.Euclidean;
   @Getter
   @Setter
   private double maxDistance = 1.0;
   @Getter
   @Setter
   private int minPointsInCluster = 1;
   @Getter
   @Setter
   private int initialK = 20;
   @Getter
   @Setter
   private int splitSize = 2;
   @Getter
   @Setter
   private boolean verbose = false;
   @Getter
   @Setter
   private int kMeansIterations = 20;
   @Getter
   @Setter
   private int maxClusterSize = Integer.MAX_VALUE;

   @Override
   public FlatClustering cluster(@NonNull MStream<NDArray> instances) {
      FlatClustering finalClustering = new FlatCentroidClustering(this, distanceMeasure);
      FlatClustering initialCluster = kmeans(instances, initialK);
      initialCluster.forEach(cluster -> doClusterRound(cluster).forEach(finalClustering::addCluster));
      for (int i = 0; i < finalClustering.size(); i++) {
         finalClustering.get(i).setId(i);
      }
      return finalClustering;
   }

   private List<Cluster> doClusterRound(Cluster cluster) {
      if (cluster.size() == 0) {
         if (verbose) {
            logInfo("Ignoring empty cluster");
         }
      } else if (cluster.size() > maxClusterSize) {
         if (verbose) {
            logInfo("Dividing with size={0} and avg. distance={1} into {2} new clusters",
                    cluster.size(),
                    cluster.getScore(),
                    splitSize);
         }
         List<Cluster> clusterList = new ArrayList<>();
         kmeans(StreamingContext.local().stream(cluster.getPoints()), splitSize)
            .forEach(c -> clusterList.addAll(doClusterRound(c)));
         return clusterList;
      } else if (cluster.size() >= minPointsInCluster && cluster.getScore() <= maxDistance) {
         if (verbose) {
            logInfo("Added cluster with size={0} and avg. distance={1}", cluster.size(), cluster.getScore());
         }
         return Collections.singletonList(cluster);
      } else if (cluster.size() >= minPointsInCluster
                    && (cluster.size() / splitSize) >= (minPointsInCluster / 2.0)) {
         if (verbose) {
            logInfo("Dividing with size={0} and avg. distance={1} into {2} new clusters",
                    cluster.size(),
                    cluster.getScore(),
                    splitSize);
         }
         List<Cluster> clusterList = new ArrayList<>();
         kmeans(StreamingContext.local().stream(cluster.getPoints()), splitSize)
            .forEach(c -> clusterList.addAll(doClusterRound(c)));
         return clusterList;
      } else if (verbose) {
         logInfo("Ignoring cluster of size={0} and avg. distance={1}",
                 cluster.size(),
                 cluster.getScore());
      }
      return Collections.emptyList();
   }

   private FlatClustering kmeans(MStream<NDArray> vectors, int K) {
      return new KMeans(K, distanceMeasure, kMeansIterations).cluster(vectors);
   }

   /**
    * Sets distance measure.
    *
    * @param distanceMeasure the distance measure
    */
   public void setDistanceMeasure(@NonNull DistanceMeasure distanceMeasure) {
      this.distanceMeasure = distanceMeasure;
   }

}//END OF DivisiveKMeans
