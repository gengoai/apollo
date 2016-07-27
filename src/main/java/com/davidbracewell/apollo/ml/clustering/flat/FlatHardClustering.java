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

import com.davidbracewell.apollo.affinity.DistanceMeasure;
import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.ml.EncoderPair;
import com.davidbracewell.apollo.ml.FeatureVector;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.clustering.Cluster;
import com.davidbracewell.apollo.ml.clustering.Clustering;
import com.davidbracewell.tuple.Tuple2;
import lombok.NonNull;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static com.davidbracewell.tuple.Tuples.$;

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
  public int hardCluster(@NonNull Instance instance) {
    Vector vector = instance.toVector(getEncoderPair());
    return clusters.parallelStream()
      .map(c -> $(c.getIndex(), getDistanceMeasure().calculate(vector, c.getCentroid())))
      .min((t1, t2) -> Double.compare(t1.getValue(), t2.getValue()))
      .map(Tuple2::getKey)
      .orElse(-1);
  }

  @Override
  public double[] softCluster(Instance instance) {
    double[] distances = new double[size()];
    Arrays.fill(distances, Double.POSITIVE_INFINITY);
    FeatureVector vector = instance.toVector(getEncoderPair());
    int assignment = hardCluster(instance);
    if (assignment >= 0) {
      distances[assignment] = getDistanceMeasure().calculate(clusters.get(assignment).getCentroid(), vector);
    }
    return distances;
  }

}//END OF FlatHardClustering
