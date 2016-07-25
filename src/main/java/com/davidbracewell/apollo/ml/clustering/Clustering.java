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

import com.davidbracewell.apollo.ApolloMath;
import com.davidbracewell.apollo.affinity.DistanceMeasure;
import com.davidbracewell.apollo.ml.EncoderPair;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.Model;
import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.stream.MPairStream;
import com.davidbracewell.tuple.Tuples;
import lombok.NonNull;

import java.util.List;


/**
 * The type Clustering.
 *
 * @author David B. Bracewell
 */
public abstract class Clustering extends Model {
  private static final long serialVersionUID = 1L;
  private final DistanceMeasure distanceMeasure;

  /**
   * Instantiates a new Clustering.
   *
   * @param encoderPair     the encoder pair
   * @param distanceMeasure the distance measure
   */
  protected Clustering(EncoderPair encoderPair, DistanceMeasure distanceMeasure) {
    super(encoderPair);
    this.distanceMeasure = distanceMeasure;
  }

  /**
   * Gets distance measure.
   *
   * @return the distance measure
   */
  public DistanceMeasure getDistanceMeasure() {
    return distanceMeasure;
  }

  /**
   * Size int.
   *
   * @return the int
   */
  public abstract int size();

  /**
   * Get cluster.
   *
   * @param index the index
   * @return the cluster
   */
  public abstract Cluster get(int index);

  /**
   * Is flat boolean.
   *
   * @return the boolean
   */
  public boolean isFlat() {
    return false;
  }

  /**
   * Is hierarchical boolean.
   *
   * @return the boolean
   */
  public boolean isHierarchical() {
    return false;
  }

  /**
   * Gets root.
   *
   * @return the root
   */
  public abstract Cluster getRoot();

  /**
   * Gets clusters.
   *
   * @return the clusters
   */
  public abstract List<Cluster> getClusters();

  /**
   * Hard cluster int.
   *
   * @param instance the instance
   * @return the int
   */
  public int hardCluster(@NonNull Instance instance) {
    return ApolloMath.argMin(softCluster(instance)).getV1();
  }

  /**
   * Soft cluster double [ ].
   *
   * @param instance the instance
   * @return the double [ ]
   */
  public abstract double[] softCluster(Instance instance);


  public MPairStream<Instance, Integer> hardCluster(@NonNull Dataset<Instance> dataset) {
    return dataset.stream().parallel()
      .mapToPair(i -> Tuples.$(i, hardCluster(i)));
  }

  public MPairStream<Instance, double[]> softCluster(@NonNull Dataset<Instance> dataset) {
    return dataset.stream().parallel()
      .mapToPair(i -> Tuples.$(i, softCluster(i)));
  }


}//END OF Clustering
