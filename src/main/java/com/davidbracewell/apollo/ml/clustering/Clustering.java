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

package com.davidbracewell.apollo.ml.clustering;

import com.davidbracewell.apollo.affinity.DistanceMeasure;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.Model;
import lombok.NonNull;


/**
 * <p>Represents the results of clustering. Is treated as a {@link Model} and allows for determining the cluster of new
 * instances.</p>
 *
 * @author David B. Bracewell
 */
public interface Clustering extends Model, Iterable<Cluster> {

   /**
    * Gets the distance measure used to create the clustering.
    *
    * @return the distance measure
    */
   DistanceMeasure getDistanceMeasure();

   /**
    * The number of clusters
    *
    * @return the number of clusters
    */
   int size();

   /**
    * Gets the  cluster for the given index.
    *
    * @param index the index
    * @return the cluster
    */
   Cluster get(int index);

   /**
    * Checks if the clustering is flat
    *
    * @return True if flat, False otherwise
    */
   boolean isFlat();

   /**
    * Checks if the clustering is hierarchical
    *
    * @return True if hierarchical, False otherwise
    */
   boolean isHierarchical();

   /**
    * Gets the root of the hierarchical cluster.
    *
    * @return the root
    */
   Cluster getRoot();

   /**
    * Performs a hard clustering, which determines the single cluster the given instance belongs to
    *
    * @param instance the instance
    * @return the index of the cluster that the instance belongs to
    */
   int hardCluster(@NonNull Instance instance);

   /**
    * Performs a soft clustering, which provides a membership probability of the given instance to the clusters
    *
    * @param instance the instance
    * @return membership probability of the given instance to the clusters
    */
   double[] softCluster(@NonNull Instance instance);


}//END OF Clustering
