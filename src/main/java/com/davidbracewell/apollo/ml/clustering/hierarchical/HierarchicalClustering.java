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


import com.davidbracewell.apollo.ml.EncoderPair;
import com.davidbracewell.apollo.ml.clustering.Cluster;
import com.davidbracewell.apollo.ml.clustering.Clustering;
import lombok.NonNull;

import java.util.ArrayList;
import java.util.List;

/**
 * The type Hierarchical clustering.
 *
 * @author David B. Bracewell
 */
public class HierarchicalClustering extends Clustering {
  private static final long serialVersionUID = 1L;
  private final Cluster root;

  /**
   * Instantiates a new Hierarchical clustering.
   *
   * @param encoderPair the encoder pair
   * @param clusters    the clusters
   * @param root        the root
   */
  public HierarchicalClustering(@NonNull EncoderPair encoderPair, @NonNull List<Cluster> clusters, @NonNull Cluster root) {
    super(encoderPair, clusters);
    this.root = root;
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
    return new Clustering(getEncoderPair(), flat);
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
