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
import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.stream.MPairStream;
import lombok.NonNull;

import java.util.List;

import static com.davidbracewell.tuple.Tuples.$;


/**
 * The type Clustering.
 *
 * @author David B. Bracewell
 */
public interface Clustering extends Model, Iterable<Cluster> {

   /**
    * Gets distance measure.
    *
    * @return the distance measure
    */
   DistanceMeasure getDistanceMeasure();


   /**
    * Size int.
    *
    * @return the int
    */
   int size();

   /**
    * Get cluster.
    *
    * @param index the index
    * @return the cluster
    */
   Cluster get(int index);

   /**
    * Is flat boolean.
    *
    * @return the boolean
    */
   boolean isFlat();

   /**
    * Is hierarchical boolean.
    *
    * @return the boolean
    */
   boolean isHierarchical();

   /**
    * Gets root.
    *
    * @return the root
    */
   Cluster getRoot();

   /**
    * Gets clusters.
    *
    * @return the clusters
    */
   List<Cluster> getClusters();

   /**
    * Hard cluster int.
    *
    * @param instance the instance
    * @return the int
    */
   int hardCluster(@NonNull Instance instance);

   /**
    * Soft cluster double [ ].
    *
    * @param instance the instance
    * @return the double [ ]
    */
   double[] softCluster(@NonNull Instance instance);


   /**
    * Hard cluster m pair stream.
    *
    * @param dataset the dataset
    * @return the m pair stream
    */
   default MPairStream<Instance, Integer> hardCluster(@NonNull Dataset<Instance> dataset) {
      return dataset.stream().parallel()
                    .mapToPair(i -> $(i, hardCluster(i)));
   }

   /**
    * Soft cluster m pair stream.
    *
    * @param dataset the dataset
    * @return the m pair stream
    */
   default MPairStream<Instance, double[]> softCluster(@NonNull Dataset<Instance> dataset) {
      return dataset.stream().parallel()
                    .mapToPair(i -> $(i, softCluster(i)));
   }


}//END OF Clustering
