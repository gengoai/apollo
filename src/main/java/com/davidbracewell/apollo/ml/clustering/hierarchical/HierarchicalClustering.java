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

package com.davidbracewell.apollo.ml.clustering.hierarchical;


import com.davidbracewell.apollo.affinity.DistanceMeasure;
import com.davidbracewell.apollo.ml.EncoderPair;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.clustering.Cluster;
import com.davidbracewell.apollo.ml.clustering.Clustering;
import com.davidbracewell.apollo.ml.clustering.flat.FlatHardClustering;
import lombok.NonNull;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * The type Hierarchical clustering.
 *
 * @author David B. Bracewell
 */
public class HierarchicalClustering extends Clustering {
  private static final long serialVersionUID = 1L;
  Cluster root;
  Linkage linkage;

  /**
   * Instantiates a new Hierarchical clustering.
   *
   * @param encoderPair the encoder pair
   */
  public HierarchicalClustering(@NonNull EncoderPair encoderPair, @NonNull DistanceMeasure distanceMeasure) {
    super(encoderPair, distanceMeasure);
  }


  /**
   * Gets root.
   *
   * @return the root
   */
  public Cluster getRoot() {
    return root;
  }

  /**
   * As flat clustering.
   *
   * @param threshold the threshold
   * @return the clustering
   */
  public Clustering asFlat(double threshold) {
    List<Cluster> flat = new ArrayList<>();
    process(root, flat, threshold);
    return new FlatHardClustering(getEncoderPair(), getDistanceMeasure(), flat);
  }

  @Override
  public boolean isHierarchical() {
    return true;
  }

  @Override
  public int size() {
    return 1;
  }

  @Override
  public Cluster get(int index) {
    return getClusters().get(index);
  }

  @Override
  public List<Cluster> getClusters() {
    return Collections.singletonList(root);
  }

  @Override
  public double[] softCluster(@NonNull Instance instance) {
    return new double[]{
      linkage.calculate(instance.toVector(getEncoderPair()), root, getDistanceMeasure())
    };
  }

  private void process(Cluster c, List<Cluster> flat, double threshold) {
    if (c == null) {
      return;
    }
    if (c.getScore() <= threshold) {
      flat.add(c);
    } else {
      process(c.getLeft(), flat, threshold);
      process(c.getRight(), flat, threshold);
    }
  }


}//END OF HierarchicalClustering
