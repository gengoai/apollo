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
import com.gengoai.apollo.stat.measure.DistanceMeasure;
import com.gengoai.guava.common.base.Preconditions;
import com.gengoai.stream.MStream;
import com.gengoai.apollo.ml.clustering.Cluster;
import com.gengoai.apollo.ml.clustering.Clusterer;
import lombok.Getter;
import lombok.NonNull;
import lombok.Setter;

import java.util.Iterator;
import java.util.List;

/**
 * <p>Implementation of the one-shot clustering algorithm, which passes over the data one time assigning each instance
 * to its closest cluster based on average distance or creating a new cluster if no existing cluster is within a
 * predefined distance.</p>
 *
 * @author David B. Bracewell
 */
public class OneShotClusterer extends Clusterer<FlatClustering> {
   private static final long serialVersionUID = 1L;
   @Getter
   private DistanceMeasure distanceMeasure;
   @Getter
   @Setter
   private double threshold;

   /**
    * Instantiates a new One shot clusterer.
    *
    * @param threshold the threshold in which the distance between an instance and cluster causes the instance to be
    *                  added to the cluster
    * @param measure   the distance measure to use.
    */
   public OneShotClusterer(double threshold, DistanceMeasure measure) {
      this.threshold = threshold;
      this.distanceMeasure = Preconditions.checkNotNull(measure);
   }

   @Override
   public FlatClustering cluster(@NonNull MStream<NDArray> instanceStream) {
      OneShotClustering clustering = new OneShotClustering(this, distanceMeasure);

      List<NDArray> instances = instanceStream.collect();
      for (NDArray ii : instances) {
         double minD = Double.POSITIVE_INFINITY;
         int minI = 0;
         for (int k = 0; k < clustering.size(); k++) {
            double d = distance(ii, clustering.get(k));
            if (d < minD) {
               minD = d;
               minI = k;
            }
         }

         if (minD <= threshold) {
            clustering.get(minI).addPoint(ii);
         } else {
            Cluster newCluster = new Cluster();
            newCluster.addPoint(ii);
            clustering.addCluster(newCluster);
         }

      }

      for (Iterator<Cluster> itr = clustering.iterator(); itr.hasNext(); ) {
         Cluster c = itr.next();
         if (c == null || c.size() == 0) {
            itr.remove();
         }
      }

      return clustering;
   }


   private double distance(NDArray ii, Cluster cluster) {
      double d = 0;
      for (NDArray jj : cluster) {
         d += distanceMeasure.calculate(ii, jj);
      }
      return d / (double) cluster.size();
   }

   /**
    * Sets the distance measure to use.
    *
    * @param distanceMeasure the distance measure
    */
   public void setDistanceMeasure(@NonNull DistanceMeasure distanceMeasure) {
      this.distanceMeasure = distanceMeasure;
   }

}//END OF OneShotClusterer
