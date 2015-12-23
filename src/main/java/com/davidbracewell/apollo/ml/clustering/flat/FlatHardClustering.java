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

import com.davidbracewell.apollo.ApolloMath;
import com.davidbracewell.apollo.affinity.DistanceMeasure;
import com.davidbracewell.apollo.ml.EncoderPair;
import com.davidbracewell.apollo.ml.FeatureVector;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.clustering.Cluster;
import com.davidbracewell.apollo.ml.clustering.Clustering;

import java.util.Collections;
import java.util.List;

/**
 * @author David B. Bracewell
 */
public class FlatHardClustering extends Clustering {
  private static final long serialVersionUID = 1L;
  List<Cluster> clusters;

  public FlatHardClustering(EncoderPair encoderPair, DistanceMeasure distanceMeasure) {
    super(encoderPair, distanceMeasure);
  }

  public FlatHardClustering(EncoderPair encoderPair, DistanceMeasure distanceMeasure, List<Cluster> clusters) {
    super(encoderPair, distanceMeasure);
    this.clusters = clusters;
  }

  @Override
  public int size() {
    return clusters.size();
  }

  @Override
  public Cluster get(int index) {
    return clusters.get(index);
  }

  @Override
  public Cluster getRoot() {
    return clusters.get(0);
  }

  @Override
  public List<Cluster> getClusters() {
    return Collections.unmodifiableList(clusters);
  }

  @Override
  public double[] softCluster(Instance instance) {
    double[] distances = new double[size()];
    FeatureVector vector = instance.toVector(getEncoderPair());
    for (int i = 0; i < distances.length; i++) {
      distances[i] = getDistanceMeasure().calculate(clusters.get(i).getCentroid(), vector);
    }
    int min = ApolloMath.argMin(distances).getV1();
    for (int i = 0; i < distances.length; i++) {
      if (i != min) {
        distances[i] = Double.POSITIVE_INFINITY;
      }
    }
    return distances;
  }

}//END OF FlatHardClustering
